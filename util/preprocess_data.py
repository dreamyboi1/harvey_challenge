import os
import glob
from math import ceil, floor, sqrt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# function definitions
def slice_images(img_path_dict, save_dir, N = 6, verbose = False):
    """
    Function that slices original images in N smaller subimages.
    Inspired by https://github.com/samdobson/image_slicer/blob/master/image_slicer/main.py but this package is based on very old version of pillow
    and doesn't run with the newest pillow version.

    Input
    -----
    img_path_dict: Python dict of training data (identifier of data: path to data)
    
    save_dir: string that identifies folder where we want to save the generated slices

    N: integer denoting how many images we slice from original image (depending on aspect ratio, actual number of subimages can be slightly lower / higher)
       (12 was a bit too high)

    verbose: boolean that can trigger print statements (helpful for debugging)

    Output
    -----
    slice_path_dict: Python dict of sliced images where each key refers to the th

    """
    slice_path_dict = {} # new dict in which we will store the location of the generated slices
    
    num_files = len(glob.glob(save_dir + "*"))

    if num_files > 0: # if save directory contains files
        if verbose:
            print(f"{save_dir} already contains {num_files} files!")
            print("Data will only be sliced after manually deleting the existing files in save_dir and re-running this function! Otherwise who knows what data you will be training on!\n")
        # fill slice path dict
        for f in glob.glob(save_dir + "*"):
            filename = os.path.basename(f)
            key = key = filename.split(".")[0]
            slice_path_dict[key] = save_dir + filename

        return slice_path_dict
    
    # if save directory empty, create slices
    for key, path in img_path_dict.items():
        # fetch image & key properties
        img = Image.open(path)
        img_w, img_h = img.size
        filetype = img.filename.split(".")[1]

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
                new_path = f"{save_dir}{new_key}.{filetype}"
                slice_path_dict[new_key] = new_path
                tile = img.crop(area)
                tile = compress_image(tile, filetype)
                if filetype == "tif":
                    tile.save(new_path, quality = 100)
                else:
                    tile.save(new_path)
                i += 1

    return slice_path_dict

def compress_image(img, filetype, new_size_ratio = 0.25):
    new_width = int(img.size[0] * new_size_ratio)
    new_height = int(img.size[1] * new_size_ratio)
    if filetype == "png": # resample masks with NEAREST
        img = img.resize((new_width, new_height), resample = Image.Resampling.NEAREST) 
    else: # resample images with BICUBIC (smoother edges)
        img = img.resize((new_width, new_height), resample = Image.Resampling.BICUBIC)
    return img

def compress_image_size(img_path_dict, save_dir, verbose = False):
    """
    Function that compresses original images (intended for validation and test data) with following objectives:
    (1) all sides divisible by 32
    (2) original aspect ratio is kept --> we can use this only when we set batch size of corresponding data to 1
                                          (otherwise error because tensor has to store different dimensions at same time)

    Input
    -----
    img_path_dict: Python dict of training data (identifier of data: path to data)
    
    save_dir: string that identifies folder where we want to save the generated slices

    verbose: boolean that can trigger print statements (helpful for debugging)

    Output
    -----
    comp_path_dict: Python dict of compressed images (identifier of data: path to data)
    """
    comp_path_dict = {} # new dict in which we will store the location of the generated slices
    
    # if save directory contains files, only generate dict with existing files
    num_files = len(glob.glob(save_dir + "*"))

    if num_files > 0: # if save directory contains files
        if verbose:
            print(f"{save_dir} already contains {num_files} files!")
            print("Data will only be sliced after manually deleting the existing files in save_dir and re-running this function! Otherwise who knows what data you will be training on!\n")
        # fill slice path dict
        for f in glob.glob(save_dir + "*"):
            filename = os.path.basename(f)
            key = filename.split(".")[0]
            comp_path_dict[key] = save_dir + filename

        return comp_path_dict

    # if save directory empty, compress
    for key, path in img_path_dict.items():
        # fetch image & key properties
        img = Image.open(path)
        img_w, img_h = img.size
        filetype = img.filename.split(".")[1]

        #img = img.resize((img_w, img_h)) # for some reason we need to get an intermediate image object to save the new version
                                        # if we try to .save the original image, the image is unreadable (???)
        # get desired image shape
        if (img_w == 4592) and (img_h == 3072):
            new_width, new_height = 1152, 768
        else:
            new_width, new_height = 1152, 864
        
        # resize images
        if filetype == "png": # resample masks with NEAREST
            img = img.resize((new_width, new_height), resample = Image.Resampling.NEAREST) 
        else: # resample images with BICUBIC (smoother edges)
            img = img.resize((new_width, new_height), resample = Image.Resampling.BICUBIC)
        
        # save image
        new_path = f"{save_dir}{key}.{filetype}"
        comp_path_dict[key] = new_path
        if filetype == "tif":
            img.save(new_path, quality = 100)
        else:
            img.save(new_path)
    
    return comp_path_dict
    

def get_transforms(config, visualize = False):
    """
    Return the transformation functions. If visualize == True, do not apply normalization and do not convert to tensor.
    """
    # get preprocessing settings from .ini
    config_pp = config["preprocessing"]
    TRAIN_MAXSIZE, TRAIN_MINSIZE = int(config_pp["TRAIN_MAXSIZE"]), int(config_pp["TRAIN_MINSIZE"])
    MEANS, STDS = [float(i) for i in config_pp["MEANS"].split(",")], [float(i) for i in config_pp["STDS"].split(",")]

    transform_train = A.Compose(
        [
            # all sides must be divisible by 32 and all images must be of same size (because we train with batch_size > 1)
            A.LongestMaxSize(max_size = TRAIN_MAXSIZE, p = 1), # rescale image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.
            A.PadIfNeeded(min_height = TRAIN_MAXSIZE, min_width = TRAIN_MINSIZE, p = 1), # slices of 4000x3000 images are a tiny bit too small (padding of these images adds 1.33% duplication)
            A.RandomCrop(height = TRAIN_MAXSIZE, width = TRAIN_MINSIZE, p = 1), # so that all images same size
            # random permutations
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.FancyPCA(p = 0.5), # augment RGB image using https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
            # augmentation
            A.CLAHE(clip_limit = 2.0, p = 1), # always apply Contrast Limited Adaptive Histogram Equalization
            A.UnsharpMask(p = 1), # always sharpen the image
            # normalize & convert data to tensor
            A.Normalize(mean=MEANS, std=STDS), # only applied to image (not the mask!)
            ToTensorV2()
        ]
    )
    
    transform_valtest = A.Compose(
        [
            A.CLAHE(clip_limit = 2.0, p = 1), # always apply Contrast Limited Adaptive Histogram Equalization
            A.UnsharpMask(p = 1), # always sharpen the image
            A.Normalize(mean = MEANS, std = STDS),
            ToTensorV2()
        ]
    )

    if visualize:
        return A.Compose(transform_train[:-2]), A.Compose(transform_valtest[:-2])

    return transform_train, transform_valtest