import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize(image, mask, transform = None, pred = False):
    fontsize = 8
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 16))
    fig.tight_layout(w_pad = 5)

    # transform image (if necessary) & define titles
    if pred == True:
        type_description = "Predicted"
    else:
        type_description = "Original"

    transform_description = ""

    if transform is not None:
        transform_description = " (transformed)"
        augments = transform(image = np.asarray(image), mask = np.asarray(mask))
        image = Image.fromarray(augments["image"])
        mask = Image.fromarray(augments["mask"])

    ax[0].set_title(f'Original Image{transform_description}', fontsize = fontsize)
    ax[1].set_title(f'{type_description} Mask{transform_description}', fontsize = fontsize)

    # plot images
    print(image.size)
    ax[0].imshow(image)
    ax[1].imshow(mask)

def evaluate_pred(org_mask, pred_mask):
    fontsize = 8
    fig, ax = plt.subplots(1, 3, figsize=(10, 16))
    fig.tight_layout(w_pad = 5)

    # compute overlap between org_mask and pred_mask
    comparison = (np.asarray(org_mask) == np.asarray(pred_mask)).astype(np.uint8)
    _, counts = np.unique(comparison, return_counts = True)
    print(f"{round(counts[1]/np.sum(counts)*100, 2)}% of all pixels correctly identified!")

    palette = np.array([[200,  70,   0],   # red
                    [  0, 180,  30]])   # green
    overlap = palette[comparison]

    # define titles
    ax[0].set_title('Original Mask', fontsize = fontsize)
    ax[1].set_title('Predicted Mask', fontsize = fontsize)
    ax[2].set_title('Overlap between Original and Predicted Mask', fontsize = fontsize)

    # plot images
    ax[0].imshow(org_mask)
    ax[1].imshow(pred_mask)
    ax[2].imshow(overlap)