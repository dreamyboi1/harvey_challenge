# import own scripts
import util.load_data as loadData

# import external packages
import os
from math import ceil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from ray import tune

def get_accuracy(outputs, mask):
    outputs = torch.argmax(outputs, dim=1)
    correct = torch.eq(outputs, mask).int()
    return float(correct.sum()) / float(correct.numel())

def get_dataloader(dataset, batch_size, num_workers = 4):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers)
    return dataloader

def train_model(config):
    # model initialisation
    class_dict = loadData.get_class_dict()
    if config["model"] == "UNet":
        model = smp.Unet(encoder_name = config["encoder"], classes = len(class_dict), activation = "softmax2d")
    elif config["model"] == "Linknet":
        model = smp.Linknet(encoder_name = config["encoder"], classes = len(class_dict), activation = "softmax2d")
    elif config["model"] == "FPN":
        model = smp.FPN(encoder_name = config["encoder"], classes = len(class_dict), activation = "softmax2d")
    elif config["model"] == "PSPNET":
        model = smp.PSPNet(encoder_name = config["encoder"], classes = len(class_dict), activation = "softmax2d")
    elif config["model"] == "DeepLabV3":
        model = smp.DeepLabV3(encoder_name = config["encoder"], classes = len(class_dict), activation = "softmax2d")
    elif config["model"] == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(encoder_name = config["encoder"], classes = len(class_dict), activation = "softmax2d")

    # how many epochs we want to train for (at maximum)
    max_epochs = int(config["max_epochs"])

    # where we want to run the model (so this code can run on cpu, gpu, multiple gpus depending on system)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # load datasets
    trainset, valset, _ = loadData.get_datasets()

    # create dataloaders
    trainloader = get_dataloader(trainset, int(config["batch_size"]))
    valloader = get_dataloader(valset, 1) # batch_size MUST be 1 because of our preprocessing (to keep original aspect ratios, val images may have different shapes)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])

    # define learning rate scheduler
    scheduler_step = None
    if config["lr_scheduler"] == "Cosine": # we start with sampled learning rate and decay it (sampled lr should be rather high probably)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_epochs, eta_min = 1e-5)
        scheduler_step = "every epoch"
    elif config["lr_scheduler"] == "Cyclic": # sampled learning rate is base lr! we go up from there to max_lr! (so sampled lr should be low for this to work)
        optimizer = torch.optim.SGD(model.parameters(), lr = config["lr"], momentum = 0.9) # CyclicLR only works with optimizer that has momentum attribute (Adam doesn't work, so we use SGD)
        steps_per_epoch = int(ceil(trainset.__len__() / config["batch_size"]))
        step_size_up = 4 * steps_per_epoch # computed according to https://arxiv.org/abs/1506.01186, constant should be anywhere between 2 to 8
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = config["lr"], max_lr = config["lr"] * 10, step_size_up = step_size_up, mode = "triangular2")
        scheduler_step = "every training batch"
    elif config["lr_scheduler"] == "OneCycle": # just like Cyclic (sampled lr should be low for this to work)
        steps_per_epoch = int(ceil(trainset.__len__() / config["batch_size"]))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = config["lr"] * 10, epochs = max_epochs, steps_per_epoch = steps_per_epoch)
        scheduler_step = "every training batch" # falsely used "each epoch" before (so learning rate slowly increased across the epochs) and that worked very well actually...

    # define criterion to compute loss
    criterion = nn.CrossEntropyLoss()

    # train and evaluate the model
    max_val_accuracy = 0
    max_val_accuracy_low_bias = 0
    low_bias_thresh = 0.01 # models with abs(val_acc - trn_acc) < thresh don't show overfit

    for epoch in range(0, max_epochs):
        ##TRAINING##
        model.train()
        trn_loss, trn_accuracy = 0.0, 0.0
        trn_steps = 0
        
        for batch in trainloader:
            # get inputs (data is a tuple of (images, masks))
            img_batch, mask_batch, _ = batch
            mask_batch = mask_batch.type(torch.LongTensor)
            img_batch, mask_batch = img_batch.to(device), mask_batch.to(device)
            
            # zero optimizer gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(img_batch).float().to(device) #model(img_batch)
            loss = criterion(outputs, mask_batch) #criterion(outputs.float().to(device), mask_batch)
            loss.backward()
            optimizer.step()
            
            # compute statistics
            trn_loss += loss.item()
            trn_accuracy += get_accuracy(outputs, mask_batch)
            trn_steps += 1

            # schedule learning rate if necessary
            if scheduler_step == "every training batch":
                scheduler.step()

        ##VALIDATION##
        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        val_steps = 0
        
        for batch in valloader:
            with torch.no_grad():
                img_batch, mask_batch, _ = batch
                mask_batch = mask_batch.type(torch.LongTensor)
                img_batch, mask_batch = img_batch.to(device), mask_batch.to(device)
                
                # compute predictions
                outputs = model(img_batch).float().to(device) #model(img_batch)
                
                # compute loss
                loss = criterion(outputs, mask_batch)
                
                # compute statistics
                val_loss += loss.item()
                val_accuracy += get_accuracy(outputs, mask_batch)
                val_steps += 1

        ##METRICS of this epoch##
        cur_lr = optimizer.param_groups[0]["lr"]
        trn_accuracy = trn_accuracy / trn_steps
        val_accuracy = val_accuracy / val_steps
        bias = abs(val_accuracy - trn_accuracy)

        ##SAVE current best model##
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            path = os.path.abspath("")+"\\best-model-overall.pt"
            torch.save(model.state_dict(), path)

        if (val_accuracy > max_val_accuracy_low_bias) and (bias < low_bias_thresh):
            max_val_accuracy_low_bias = val_accuracy
            path = os.path.abspath("")+"\\best-model-unbiased.pt"
            torch.save(model.state_dict(), path)

        ##REPORT##
        tune.report(loss = (val_loss / val_steps), trn_acc = trn_accuracy, val_acc=val_accuracy, max_val_acc = max_val_accuracy, bias = bias, cur_lr = cur_lr)
        # print(f"Train accuracy: {round(trn_accuracy, 3)}")
        # print(f"Val accuracy: {round(val_accuracy, 3)}\n")

        ##SCHEDULE learning rate if necessary##
        if scheduler_step == "every epoch":
            scheduler.step()

    print("Finished Training")

def predict(model, dataset, save_dir, test = True):

    y_test_paths = {} # create dict in which we will store the paths to the predictions

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # create dataloaders
    dataloader = get_dataloader(dataset, 1)

    # get shapes dict from dataloader to resize images properly
    if type(dataloader.dataset) == torch.utils.data.dataset.Subset:
        shapes = dataloader.dataset.dataset.shapes
    else:
        shapes = dataloader.dataset.shapes

    # make predictions
    model.eval()

    for batch in dataloader:
        if test == True:
            img_batch, names_batch = batch
        else:
            img_batch, _, names_batch = batch
        img_batch = img_batch.to(device)

        # compute predictions
        outputs = model(img_batch)
        
        # save predictions as masks
        with torch.no_grad():
            preds = torch.argmax(outputs.to(device), dim=1)
            for name, pred in zip(names_batch, preds):
                img = Image.fromarray(pred.cpu().numpy().astype(np.uint8)) # convert Image data to PIL format
                size = shapes[name] # fetch original shape of image
                img = img.resize(size = size, resample = Image.Resampling.NEAREST) # resize mask to size of original image
                path = save_dir + name + ".png" # where we want to save the image
                y_test_paths[name] = path # save storage path
                img.save(path, format = "PNG") # save image

    return y_test_paths