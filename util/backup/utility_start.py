# parse & handle data
import os
import glob
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def get_notebook_path():
    return os.path.abspath("")

def get_class_dict(path_metadata):
    """
    Get the class labels and corresponding names within a Python dict.
    
    Input
    -----
    metadata: path (String) to provided metadata.json file (from downloaded data)
    
    Ouput
    -----
    class_dict: Python dict with class value as key and class label as value (e.g. 0: "Property Roof")
    """
    # fetch metadata file
    fin = open(path_metadata, 'r')
    metadata = json.load(fin)
    fin.close()
    # create class_dict
    class_dict = {}
    class_dict[0] = "background"
    for idx, val in enumerate(metadata["label:metadata"][0]["options"]):
        class_dict[idx+1] = val # background class has value 0
        
    return class_dict

def get_train_test_doc_paths(X_path, y_path, notebook_path):
    """
    Function that retrieves the global paths (as string )
    
    Input
    -----
    X_path: string that identifies folder of input data (the .tif images in the raw folder)
    
    y_path: string that identifies folder of output data (.png masks are only available for training data)
    
    notebook_path: directory of this notebook (as string)
    
    
    Output
    -----
    X_train_paths: Python dict of training data where each key refers to the input_path
    
    y_train_paths: Python dict of training data where each key refers to the output_path  
    
    X_test_paths: Python dict of test data where each key refers to the input_path
    
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
            
    print(f"Number of images for training: {len(X_train_paths)}")
    print(f"Number of images for test: {len(X_test_paths)}")
            
    return X_train_paths, y_train_paths, X_test_paths


def get_data_as_np_array(paths, desired_shape, return_shapes = True):
    """
    Retrieve image / mask data based on a Python dict of the corresponding paths.
    ATTENTION: this function reads the whole dataset into memory! When desired_shape is large, this 
    can overload the RAM!
    
    Input
    -----
    paths: Python dict of paths of the image / mask data that should be retrieved (key is image name, value is path)
    
    desired_shape: tuple of int (width, height) that denotes the desired shape of the image

    return_shapes: boolean that controls whether we want to return the shape dict
    
    """
    shapes = {}
    data = []
    
    for key, path in paths.items():
        # retrieve image and save its size
        img = Image.open(path)
        shapes[key] = img.size
        
        # if image not of desired shape, resize it
        if img.size != desired_shape:
            img = img.resize(desired_shape, resample = Image.Resampling.NEAREST)
        
        data.append(np.asarray(img)) # save image (of desired shape)
        
    data = np.stack(data, axis = 0) # convert list of numpy arrays into numpy array

    if return_shapes:
        return shapes, data
    
    return data

class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms. Inspired by https://9to5answer.com/pytorch-transforms-on-tensordataset.
    """
    def __init__(self, tensors, targets = True, transform=None, target_transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)

        if self.targets == False: # if this Dataset has no targets
            return x
        
        # if Dataset has targets
        y = self.tensors[1][index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        if self.targets == False: # if this Dataset has no targets
            return self.tensors.size(0)
        return self.tensors[0].size(0)

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
                img = self.transform(image = img)
            return img

        # if we have masks available, transform image & mask at same time
        y_path = self.mask_paths[key]
        mask = np.asarray(Image.open(y_path))

        if self.transform:
            augments = self.transform(image = img, mask = mask) # non-spatial transformations will only be applied to the image (e.g. RandomBrightness)
            img = augments["image"]
            mask = augments["mask"]
        
        return img, mask

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