# parse & handle data
import os
import glob
import json
from math import ceil, floor, sqrt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from ray import tune

def get_notebook_path():
    return os.path.abspath("")
    
def visualize(image, mask, transform = None, pred = False):
    fontsize = 8
    fig, ax = plt.subplots(1, 2, figsize=(8, 16), squeeze=True)
    if transform is not None:
        augments = transform(image = np.asarray(image), mask = np.asarray(mask))
        image = Image.fromarray(augments["image"])
        mask = Image.fromarray(augments["mask"])
        ax[0].set_title('Original Image (transformed)', fontsize = fontsize)
        if pred == True:
            ax[1].set_title('Predicted Mask', fontsize = fontsize)
        else:
            ax[1].set_title('Original Mask (transformed)', fontsize = fontsize)
    else:
        ax[0].set_title('Original Image', fontsize = fontsize)
        if pred == True:
            ax[1].set_title('Predicted Mask (upsampled to original resolution)', fontsize = fontsize)
        else:
            ax[1].set_title('Original Mask', fontsize = fontsize)
    ax[0].imshow(image)
    ax[1].imshow(mask)


class DataGetter():
    """
    This class retrieves
    (1) the class values and labels
    (2) data for pre-training
    (3) data for training & validation
    (4) data for test
    """
    def __init__(self): 
        pass

    def get_class_dict(self, path):
        """
        Get the class labels and corresponding names within a Python dict.
        
        Input
        -----
        path: path (String) to provided metadata.json file (from downloaded data)
        
        Ouput
        -----
        class_dict: Python dict with class value as key and class label as value (e.g. 0: "Property Roof")
        """
        # fetch metadata file
        fin = open(path, 'r')
        metadata = json.load(fin)
        fin.close()
        # create class_dict
        class_dict = {}
        class_dict[0] = "Background"
        for idx, val in enumerate(metadata["label:metadata"][0]["options"]):
            class_dict[idx+1] = val # background class has value 0
            
        return class_dict

    def get_train_test_doc_paths(self, X_path, y_path):
        """
        Function that retrieves the global paths (as string )
        
        Input
        -----
        X_path: string that identifies folder of input data (the .tif images in the raw folder)
        
        y_path: string that identifies folder of output data (.png masks are only available for training data)
        
        Output
        -----
        X_train_paths: Python dict of training data (identifier of data: path to data)
        
        y_train_paths: Python dict of training data (identifier of data: path to data)
        
        X_test_paths: Python dict of test data (identifier of data: path to data)
        
        """  
        X_train_paths, y_train_paths, X_test_paths = {}, {}, {}

        for f in glob.glob(X_path + '*.tif'):
            
            X_filename = os.path.basename(f) # get filename (e.g. 6411.tif) of data
            y_filename = X_filename[:-4] + '.png' # get filename of corresponding mask (e.g. 6411.png)
            
            key = X_filename[:-4] # key that we will insert in the dicts
            
            if os.path.exists(y_path + y_filename): # if this file has a mask, it's training data
                X_train_paths[key] = X_path + X_filename
                y_train_paths[key] = y_path + y_filename
                
            else: # otherwise test data
                X_test_paths[key] = X_path + X_filename
                
        return X_train_paths, y_train_paths, X_test_paths

    def slice_images(self, img_path_dict, save_dir, N):
        """
        Function that slices original images in N smaller subimages.
        Inspired by https://github.com/samdobson/image_slicer/blob/master/image_slicer/main.py but this package is based on very old version of pillow
        and doesn't run with the newest pillow version.

        Input
        -----
        img_path_dict: Python dict of training data (identifier of data: path to data)
        
        save_dir: string that identifies folder where we want to save the generated slices

        N: integer denoting how many images we slice from original image (depending on aspect ratio, actual number of subimages can be slightly lower / higher)

        Output
        -----
        slice_path_dict: Python dict of sliced images where each key refers to the th

        """
        slice_path_dict = {} # new dict in which we will store the location of the generated slices
        
        num_files = len(glob.glob(save_dir + "*"))

        if num_files > 0: # if save directory contains files
            print(f"{save_dir} already contains {num_files} files!")
            print("Data will only be sliced after manually deleting the existing files in save_dir and re-running this function! Otherwise who knows what data you will be training on!\n")
            # fill slice path dict
            for f in glob.glob(save_dir + "*"):
                filename = os.path.basename(f)
                key = filename[:-4]
                slice_path_dict[key] = save_dir + "\\" + filename

            return slice_path_dict
        
        # if save directory empty, create slices
        for key, path in img_path_dict.items():
            # fetch image & key properties
            img = Image.open(path)
            img_w, img_h = img.size
            filetype = img.filename[-4:]

            # compute size of slices
            cols = int(ceil(sqrt(N)))
            rows = int(ceil(N / float(cols)))
            tile_w, tile_h = int(floor(img_w / cols)), int(floor(img_h / rows))

            # create & save slices
            i = 1
            for w in range(0, img_w - cols, tile_w):
                for h in range(0, img_h - rows, tile_h):
                    area = (w, h, w + tile_w, h + tile_h)
                    new_key = f"{key}_{i}"
                    new_path = f"{save_dir}\\{new_key}{filetype}"
                    slice_path_dict[new_key] = new_path
                    tile = img.crop(area)
                    tile.save(new_path)
                    i += 1

        return slice_path_dict

    def get_datasets(self, X_train_paths, y_train_paths, X_test_paths, transform_train, transform_test, slice_training_data = (True, 12), VAL_PERCENTAGE = 0.1, seed = 42):
        """
        Function that retrieves train, validation and test datasets (of type CustomDataset, defined below)
        
        Input
        -----
        X_train_paths: Python dict of training data (identifier of data: path to data)
        
        y_train_paths: Python dict of training data (identifier of data: path to data)
        
        X_test_paths: Python dict of test data (identifier of data: path to data)

        transform_train: Albumentations.Compose object that defines transforms on training & validation data

        transform_test: Albumentations.Compose object that defines transforms on test data

        val_percentage: Share of training data to be used for validation

        Output
        -----
        trainset, valset, testset: Instances of CustomDataset with respective data passed to it
        
        """        
        # compute sizes of train & validation datasets
        N = len(X_train_paths) # number of overall samples
        VAL_SIZE = int(VAL_PERCENTAGE*N)
        TRN_SIZE = N - VAL_SIZE

        # fix generator for train validation split (for reproducible results)
        generator = torch.Generator().manual_seed(seed) 

        # create training + validation pytorch datasets 
        raw_trainset = CustomDataset(X_train_paths, y_train_paths, transform = transform_train)
        trainset, valset = torch.utils.data.random_split(raw_trainset, [TRN_SIZE, VAL_SIZE], generator = generator)

        # create test pytorch dataset (no masks available)
        testset = CustomDataset(X_test_paths, transform = transform_test)

        return trainset, valset, testset

class CustomDataset(Dataset):
    """
    Dataset with input as directories (so we don't have to load the whole Data in memory
    instanciate the Dataset --> with CustomTensorDataset we have to do that)
    Attention: img_paths and mask_paths are of input type dict!
    """
    def __init__(self, img_paths, mask_paths=None, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

        # save original image sizes in shapes dict (they can be different!)
        self.shapes = {}
        for key, path in self.img_paths.items():
            # retrieve image and save its size
            img = Image.open(path)
            self.shapes[key] = img.size

    def __getitem__(self, index):
        # fetch key to search in dicts
        key = list(self.img_paths.keys())[index]

        # fetch image with key
        x_path = self.img_paths[key]
        img = np.asarray(Image.open(x_path))

        # if no masks, apply transformation & we're done
        if self.mask_paths is None: # when we have no masks
            if self.transform:
                img = self.transform(image = img)["image"]
            return img, key

        # if we have masks available, transform image & mask at same time
        y_path = self.mask_paths[key]
        mask = np.asarray(Image.open(y_path))

        if self.transform:
            augments = self.transform(image = img, mask = mask) # non-spatial transformations will only be applied to the image (e.g. RandomBrightness)
            img = augments["image"]
            mask = augments["mask"]
        
        return img, mask, key

    def __len__(self):
        return len(self.img_paths)

    def compute_mean_std(self, indices = None):
        # define iterator
        if indices is not None:
            iterator = np.array(list(self.img_paths.values()))[indices]
        else:
            iterator = range(0, self.__len__())
        
        # initialise values for computation
        R_sum, G_sum, B_sum = 0, 0, 0
        R_sqsum, G_sqsum, B_sqsum = 0, 0, 0
        count = 0

        # iterate over data
        for path in iterator:
            img = np.asarray(Image.open(path)).astype(np.float32)/255
            
            R_sum += np.sum(img[:,:,0])
            G_sum += np.sum(img[:,:,1])
            B_sum += np.sum(img[:,:,2])

            R_sqsum += np.sum(img[:,:,0] ** 2)
            G_sqsum += np.sum(img[:,:,1] ** 2)
            B_sqsum += np.sum(img[:,:,2] ** 2)

            count += img.shape[0] * img.shape[1] # count how many pixels we have added to each channel within this step

        # compute mean
        R_mean = R_sum / count
        G_mean = G_sum / count
        B_mean = B_sum / count

        # compute std = sqrt(E[X^2] - (E[X])^2)
        R_std = (R_sqsum / count - R_mean ** 2) ** 0.5
        G_std = (G_sqsum / count - G_mean ** 2) ** 0.5
        B_std = (B_sqsum / count - B_mean ** 2) ** 0.5

        return [R_mean, G_mean, B_mean], [R_std, G_std, B_std]

class ModelPipeline():
    def __init__(self, trainset, valset, testset, class_dict, num_workers, max_epochs, criterion):
        """
        Constructor of ModelPipeline Class
        
        Input
        -----
        model: pytorch model that we want to train

        trainset, valset, testset: Instances of CustomDataset with respective data passed to it

        num_workers: int -- number of workers in dataloaders
        
        max_epochs: int -- number of epochs the model should be trained
        
        criterion: loss function object from pytorch

        """
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.class_dict = class_dict
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.criterion = criterion

    def get_accuracy(self, outputs, mask):
        outputs = torch.argmax(outputs, dim=1)
        correct = torch.eq(outputs, mask).int()
        return float(correct.sum()) / float(correct.numel())

    def train_model(self, config):
        # where we want to run the model (so this code can run on cpu, gpu, multiple gpus depending on system)
        
        if config["model"] == "UNet":
            model = smp.Unet(encoder_name = config["encoder"], classes = len(self.class_dict), activation = "softmax2d")
        elif config["model"] == "Linknet":
            model = smp.Linknet(encoder_name = config["encoder"], classes = len(self.class_dict), activation = "softmax2d")
        elif config["model"] == "FPN":
            model = smp.FPN(encoder_name = config["encoder"], classes = len(self.class_dict), activation = "softmax2d")
        elif config["model"] == "PSPNET":
            model = smp.PSPNet(encoder_name = config["encoder"], classes = len(self.class_dict), activation = "softmax2d")

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        model.to(device)

        # create dataloaders
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size = int(config["batch_size"]),
            shuffle = True,
            num_workers = int(self.num_workers))
        valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size = int(config["batch_size"]),
            shuffle = True,
            num_workers = int(self.num_workers))

        # define optimizer
        optimizer = torch.optim.Adam(params = model.parameters(), lr = config["lr"])

        # define learning rate scheduler
        if config["lr_scheduler"] == "Cosine": # we start with sampled learning rate and decay it (sampled lr should be rather high probably)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.max_epochs, eta_min = 1e-5) 
        elif config["lr_scheduler"] == "Cyclic": # sampled learning rate is base lr! we go up from there to max_lr! (so sampled lr should be low for this to work)
            steps_per_epoch = int(ceil(self.trainset.__len__() / config["batch_size"]))
            step_size_up = 2 * steps_per_epoch # computed according to https://arxiv.org/abs/1506.01186
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = config["lr"], max_lr = config["lr"] * 10, step_size_up = step_size_up, mode = "triangular2")
        elif config["lr_scheduler"] == "OneCycle": # just like Cyclic (sampled lr should be low for this to work)
            steps_per_epoch = int(ceil(self.trainset.__len__() / config["batch_size"]))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = config["lr"] * 10, epochs = self.max_epochs, steps_per_epoch = steps_per_epoch)
        # train and evaluate the model
        max_val_accuracy = 0
        for epoch in range(0, self.max_epochs):
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
                loss = self.criterion(outputs, mask_batch)    #self.criterion(outputs.float().to(device), mask_batch)
                loss.backward()
                optimizer.step()
                
                # compute statistics
                trn_loss += loss.item()
                trn_accuracy += self.get_accuracy(outputs, mask_batch)
                trn_steps += 1

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
                    loss = self.criterion(outputs, mask_batch)
                    
                    # compute statistics
                    val_loss += loss.item()
                    val_accuracy += self.get_accuracy(outputs, mask_batch)
                    val_steps += 1

            ##METRICS of this epoch##
            cur_lr = optimizer.param_groups[0]["lr"]
            trn_accuracy = trn_accuracy / trn_steps
            val_accuracy = val_accuracy / val_steps

            ##SAVE current best model##
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                path = os.path.abspath("")+"\\best-model.pt"
                torch.save(model.state_dict(), path)

            ##REPORT##
            tune.report(loss = (val_loss / val_steps), trn_acc = trn_accuracy, val_acc=val_accuracy, max_val_acc = max_val_accuracy, cur_lr = cur_lr)
            # print(f"Train accuracy: {round(trn_accuracy, 3)}")
            # print(f"Val accuracy: {round(val_accuracy, 3)}\n")

            ##SCHEDULE learning rate##
            if config["lr_scheduler"] is not None:
                scheduler.step()

        print("Finished Training")

    def predict(self, model, dataset, save_dir, test = True):

        y_test_paths = {} # create dict in which we will store the paths to the predictions

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        model.to(device)

        # create dataloaders
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 1, # batch size for predictions 1 (to be able to use higher image size)
            shuffle = True,
            num_workers = int(self.num_workers))

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