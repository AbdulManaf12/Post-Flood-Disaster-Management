import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def display_images_with_masks(image, mask, predicted=None):
    
    class_map= {'Background':0, 'Building-flooded':1, 'Building-non-flooded':2, 'Road-flooded':3, 'Road-non-flooded':4, 'Water':5, 'Tree':6, 'Vehicle':7, 'Pool':8, 'Grass':9}

    color_map = {
        "Background": [0, 0, 0],
        "Building-flooded": [255, 0, 0],
        "Building-non-flooded": [0, 255, 0],
        "Road-flooded": [0, 255, 120],
        "Road-non-flooded": [0, 0, 255],
        "Water": [255, 0, 255],
        "Tree": [70, 70, 70],
        "Vehicle": [102, 102, 156],
        "Pool": [190, 153, 153],
        "Grass": [180, 165, 180]
    }

    handles = [
        Rectangle((0, 0), 1, 1, color=np.array(c)/255) for n, c in color_map.items()
    ]
    labels = [n for n, c in color_map.items()]

    no_of_subplots = 2 if predicted is None else 3
    if predicted is None:
        plt.figure(figsize=(6, 6))
        plt.subplots_adjust(wspace=0.3) 
    else:
        plt.figure(figsize=(12, 6))

    plt.subplot(1, no_of_subplots, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    plt.title('Image')

    plt.subplot(1, no_of_subplots, 2)
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    for class_name, class_idx in class_map.items():
        color = color_map[class_name]
        mask_indices = np.where(mask == class_idx)
        mask_colored[mask_indices[0], mask_indices[1], :] = color
    plt.imshow(cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB))
    plt.title('Ground-Truth Mask')

    if predicted is not None:
        plt.subplot(1, 3, 3)
        mask_colored1 = np.zeros_like(image, dtype=np.uint8)
        for class_name1, class_idx1 in class_map.items():
            color1 = color_map[class_name1]
            mask_indices1 = np.where(predicted == class_idx1)
            mask_colored1[mask_indices1[0], mask_indices1[1], :] = color1
        plt.imshow(cv2.cvtColor(mask_colored1, cv2.COLOR_BGR2RGB))
        plt.title('Predicted Mask')

    plt.legend(handles, labels, bbox_to_anchor =(-0.8,-0.5), loc='lower center', ncol=5)
    plt.show()