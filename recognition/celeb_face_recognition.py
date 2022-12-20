from pathlib import Path
import argparse

import requests

import torch
import torchvision.transforms as T
from recognition.model import LitModel

from PIL import Image

from typing import Union

import wandb

MODEL_DIRNAME = Path(__file__).resolve().parents[1] / 'artifacts' / 'model-1i41aj0m-v11'
MODEL_FILE = 'model.ckpt'

wandb.login(key = '9a9757d2d6396045bd78ef694656af43fc46275d')
api = wandb.Api()


class CelebFaceRecognition:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_DIRNAME / MODEL_FILE 
        
        if not model_path.is_file():
            artifact = api.artifact('phum19011/celeb-face/model-1i41aj0m:v11', type='model')
            artifact_dir = artifact.download()
            model_path = Path(artifact_dir) / 'model.ckpt'

        self.model = LitModel.load_from_checkpoint(model_path) 
        self.model.eval()
        self.mapping_class2account = self.model.mapping_class2account
        self.mapping_account2name = self.model.mapping_account2name
        self.transforms = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]):
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = Image.open(image)
        
        image_tensor = self.transforms(image_pil).unsqueeze(0)
        y_pred = self.model(image_tensor).argmax(1)[0].item()
        pred_account = self.mapping_class2account[y_pred] 
        pred_name = self.mapping_account2name[pred_account] 
        return pred_name, pred_account


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename',
        type=str,
        help='path of image file'
    )
    args = parser.parse_args()
    celeb_face = CelebFaceRecognition()
    pred_name, pred_account = celeb_face.predict(args.filename)
    return pred_name, pred_account 

if __name__ == '__main__':
    main()