# import own scripts
import util.preprocess_data as prepData

# import external packages
import configparser
import os
import glob
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# get paths from .ini
config = configparser.ConfigParser()
 # had to do it this way, otherwise raytune didn't work (because scripts were executed from a different root directory)
 # before I just called "util\config.ini"
config.read('/Users/oskar/Documents/DSBA/M2/FDL/5_group project-2/util/config_oskar.ini')
config_r = config["read_paths"]
meta_path = config_r["meta_path"]
X_path, y_path = config_r["X_path"], config_r["y_path"]
config_w = config["write_paths"]
X_train_prep_path, y_train_prep_path = config_w["X_train_prep_path"], config_w["y_train_prep_path"]
X_val_prep_path, y_val_prep_path = config_w["X_val_prep_path"], config_w["y_val_prep_path"]
X_test_prep_path = config_w["X_test_prep_path"]

# function & Class definitions
def get_notebook_path():
    return os.path.abspath("")

def get_class_dict():
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
    fin = open(meta_path, 'r')
    metadata = json.load(fin)
    fin.close()
    # create class_dict
    class_dict = {}
    class_dict[0] = "Background"
    for idx, val in enumerate(metadata["label:metadata"][0]["options"]):
        class_dict[idx+1] = val # background class has value 0
        
    return class_dict

def get_train_val_test_doc_paths(VAL_PERCENTAGE = 0.1, seed = 42):
    """
    Function that retrieves in seperate Python dicts the training and test data.
    
    Output
    -----
    X_train_paths, y_train_paths: Python dicts of training data (identifier of data: path to image (X) / path to mask (y))
    
    X_val_paths, y_val_paths: Python dicts of validation data (identifier of data: path to image (X) / path to mask (y))
    
    X_test_paths: Python dict of test data (identifier of data: path to image (X))
    """  
    X_trainval_paths, y_trainval_paths, X_test_paths = {}, {}, {}

    for f in glob.glob(X_path + '*.tif'):
        
        X_filename = os.path.basename(f) # get filename (e.g. 6411.tif) of data
        key = X_filename.split(".")[0] # key that we will insert in the dicts (e.g. 6411)
        y_filename = key + '.png' # get filename of corresponding mask (e.g. 6411.png)
        
        if os.path.exists(y_path + y_filename): # if this file has a mask, it's training / validation data
            X_trainval_paths[key] = X_path + X_filename
            y_trainval_paths[key] = y_path + y_filename
            
        else: # otherwise test data
            X_test_paths[key] = X_path + X_filename

    # seperate training and validation data in different dicts
    # we want to do this stratified, meaning we want our split to have the same share
    # of 4000x3000 vs 4592x3072 images as in the overall trainval data
    X_train_paths, y_train_paths, X_val_paths, y_val_paths = {}, {}, {}, {}
    
    trainval_keys = list(X_trainval_paths.keys())

    trainval_shapes = []
    for key in trainval_keys:
        img = Image.open(X_trainval_paths[key])
        trainval_shapes.append(img.size)

    train_keys, val_keys = train_test_split(trainval_keys, 
                                            test_size = VAL_PERCENTAGE,
                                            random_state = seed,
                                            shuffle = True,
                                            stratify = trainval_shapes)

    X_train_paths, y_train_paths = {key: X_trainval_paths[key] for key in train_keys}, {key: y_trainval_paths[key] for key in train_keys}
    X_val_paths, y_val_paths = {key: X_trainval_paths[key] for key in val_keys}, {key: y_trainval_paths[key] for key in val_keys}
            
    return X_train_paths, y_train_paths, X_val_paths, y_val_paths, X_test_paths

def get_datasets():
    # get paths to training & test data
    X_train_paths, y_train_paths, X_val_paths, y_val_paths, X_test_paths = get_train_val_test_doc_paths()
    # preprocess training data
    X_train_prep_paths = prepData.slice_images(X_train_paths, X_train_prep_path)
    y_train_prep_paths = prepData.slice_images(y_train_paths, y_train_prep_path)
    # preprocess validation & test data (with this setup, we MUST use batch_size = 1 for these datasets)
    X_val_prep_paths  = prepData.compress_image_size(X_val_paths, X_val_prep_path)
    y_val_prep_paths  = prepData.compress_image_size(y_val_paths, y_val_prep_path)
    X_test_prep_paths = prepData.compress_image_size(X_test_paths, X_test_prep_path)
    # get transformations
    transform_train, transform_valtest = prepData.get_transforms(config)
    # get datasets
    trainset = CustomDataset(X_train_prep_paths, mask_paths = y_train_prep_paths, transform = transform_train, original_img_paths = X_train_paths)
    valset   = CustomDataset(X_val_prep_paths, mask_paths = y_val_prep_paths, transform = transform_valtest, original_img_paths = X_val_paths)
    testset  = CustomDataset(X_test_prep_paths, transform = transform_valtest, original_img_paths = X_test_paths)
    
    return trainset, valset, testset

class CustomDataset(Dataset):
    """
    Dataset with input as directories (so we don't have to load the whole Data in memory
    instanciate the Dataset --> with CustomTensorDataset we have to do that)
    Attention: img_paths and mask_paths are of input type dict!
    """
    def __init__(self, img_paths, mask_paths=None, transform=None, original_img_paths = None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.original_img_paths = original_img_paths

        # save original image sizes in shapes dict
        self.shapes = {}
        if original_img_paths is None:
            for key, path in self.img_paths.items():
                # retrieve image and save its size
                img = Image.open(path)
                self.shapes[key] = img.size
        else:
            for key, path in self.original_img_paths.items():
                # retrieve image and save its size
                img = Image.open(path)
                self.shapes[key] = img.size

    def __getitem__(self, index):
        # fetch key to search in dicts
        key = list(self.img_paths.keys())[index]

        # fetch image with key
        x_path = self.img_paths[key]
        img = np.array(Image.open(x_path))

        # if no masks, apply transformation & we're done
        if self.mask_paths is None: # when we have no masks
            if self.transform:
                img = self.transform(image = img)["image"]
            return img, key

        # if we have masks available, transform image & mask at same time
        y_path = self.mask_paths[key]
        mask = np.array(Image.open(y_path))

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