import torch
import os
import argparse
import evaluate

from custom_datasets import get_images, get_dataset, get_data_loaders
from model import load_model
from config import ALL_CLASSES, LABEL_COLORS_LIST
from engine import train, validate
from utils import SaveBestModel, SaveBestModelIOU

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

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
args = parser.parse_args()
print(args)

def test(model, test_dataloader, device, processor, metric):
    model.eval()
    with torch.no_grad():
        test_loss, test_miou = evaluate(
            model, test_dataloader, device, processor=processor, metric=metric
        )
    return test_loss, test_miou

if __name__ == '__main__':
    # Load best model
    best_model_path = "outputs/best_model.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, processor = load_model(num_classes=len(ALL_CLASSES))
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    # Load test dataset
    test_images, test_masks = get_images(root_path='input/road_seg/test')
    test_dataset = get_dataset(
        test_images, 
        test_masks,
        test_images,  # Assuming no labels for test set
        test_masks,   # Assuming no labels for test set
        ALL_CLASSES,
        ALL_CLASSES,
        LABEL_COLORS_LIST,
        img_size=args.imgsz,
        feature_extractor=processor,
        mode='test'
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    metric = evaluate.load("mean_iou")

    # Testing
    test_loss, test_miou = test(model, test_dataloader, device, processor, metric)
    print(f"Test Loss: {test_loss:.4f}, Test mIOU: {test_miou:.4f}")
    print('TESTING COMPLETE')
