import torch
import argparse
import evaluate

from transformers import (
    Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
)

from custom_datasets import get_images, get_dataset, get_data_loaders
from model import load_model
from config import ALL_CLASSES, LABEL_COLORS_LIST
from engine import validate

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch',
    default=4,
    help='batch size for data loader',
    type=int
)
parser.add_argument(
    '--imgsz', 
    default=[512, 416],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='outputs/final_model/' 
)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    processor = Mask2FormerImageProcessor()
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model) 
    model.to(device).eval()
    
    test_images, test_masks = get_images(root_path='input/road_seg', test=True)
    test_dataset = get_dataset(
        None,
        None, 
        None,
        None,   
        ALL_CLASSES,
        ALL_CLASSES,
        LABEL_COLORS_LIST,
        img_size=args.imgsz,
        feature_extractor=processor,
        test=True,
        test_image_paths=test_images, 
        test_mask_paths=test_masks,
    )

    test_dataloader = get_data_loaders(
        None, 
        None,
        args.batch,
        processor,
        test=True,
        test_dataset=test_dataset
    )

    for batch_idx, (images, masks) in enumerate(test_dataloader):
        print(f"Batch {batch_idx}:")
        print("Images shape:", images.shape)
        print("Masks shape:", masks.shape)
        if batch_idx == 0:
            print("First image:", images[0])
            print("First mask:", masks[0])
        if batch_idx == 2:
            break

    metric = evaluate.load("mean_iou")

    # Testing
    valid_epoch_loss, test_miou = validate(
        model,
        test_dataloader,
        device,
        ALL_CLASSES,
        LABEL_COLORS_LIST,
        0, 
        save_dir=None,
        processor=processor,
        metric=metric
    )
    print(f"Test mIOU: {test_miou:.4f}")
    print('TESTING COMPLETE')
