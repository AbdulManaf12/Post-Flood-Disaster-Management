from transformers import (
    Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
)

def load_model(backbone='facebook/mask2former-swin-tiny-ade-semantic', num_classes=1):
    image_processor = Mask2FormerImageProcessor(ignore_index=255, reduce_labels=True)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        backbone,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    return model, image_processor

#  'facebook/mask2former-swin-tiny-ade-semantic'
#  'facebook/mask2former-swin-large-ade-semantic'
# https://huggingface.co/facebook?search_models=mask2former-swin-large