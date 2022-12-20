import torch
import torch.nn as nn
import pytorch_lightning as pl
from facenet_pytorch import InceptionResnetV1

import torchmetrics
import json

from typing import List


class LitModel(pl.LightningModule):
    def __init__(self, mapping_dir: str, lr: float = 1e-4):
        super().__init__()
        with open(mapping_dir) as file:
            self.mapping_account2name = json.loads(file.read())
        self.mapping_class2account = {i: k for i, (k, v) in enumerate(self.mapping_account2name.items())} 
        self.mapping_account2class = {v: k for k, v in self.mapping_class2account.items()}

        self.n_classes = len(self.mapping_account2class)

        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=self.n_classes) 
        self.save_hyperparameters()

        self.loss = nn.CrossEntropyLoss()

        self.lr = lr


        self.init_metrics()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x): # [B, C, H, W]
        outs = self.backbone(x)
        return outs
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = self._account2class(account)
        out = self.backbone(x)
        self.cal_metrics(y, out, task='train')
        loss = self.loss(out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y = self._account2class(account)
        out = self.backbone(x)
        self.cal_metrics(y, out, task='val')
        loss = self.loss(out, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # y = self._account2class(account)
        out = self.backbone(x)
        self.cal_metrics(y, out, task='test')
        loss = self.loss(out, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def cal_metrics(self, y_true, y_pred, task='train'):
        y_pred = y_pred.argmax(1)

        eval(f'self.{task}_acc(y_pred, y_true)')
        eval(f'self.{task}_precision(y_pred, y_true)')
        eval(f'self.{task}_recall(y_pred, y_true)')
        eval(f'self.{task}_f1score(y_pred, y_true)')

        self.log(f'{task}_acc', eval(f'self.{task}_acc'), on_step=True, on_epoch=True)
        self.log(f'{task}_precision', eval(f'self.{task}_precision'), on_step=True, on_epoch=True)
        self.log(f'{task}_recall', eval(f'self.{task}_recall'), on_step=True, on_epoch=True)
        self.log(f'{task}_f1score', eval(f'self.{task}_f1score'), on_step=True, on_epoch=True)
    
    def init_metrics(self):
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes, average='macro')
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes, average='macro')
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.n_classes, average='macro')

        self.train_precision = torchmetrics.Precision(task='multiclass', num_classes=self.n_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task='multiclass', num_classes=self.n_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task='multiclass', num_classes=self.n_classes, average='macro')
    
        self.train_recall = torchmetrics.Recall(task='multiclass', num_classes=self.n_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task='multiclass', num_classes=self.n_classes, average='macro')
        self.test_recall = torchmetrics.Recall(task='multiclass', num_classes=self.n_classes, average='macro')

        self.train_f1score = torchmetrics.F1Score(task='multiclass', num_classes=self.n_classes, average='macro')
        self.val_f1score = torchmetrics.F1Score(task='multiclass', num_classes=self.n_classes, average='macro')
        self.test_f1score = torchmetrics.F1Score(task='multiclass', num_classes=self.n_classes, average='macro')

    def _class2account(self, cls: List) -> List:
        return [self.mapping_class2account[c] for c in cls]
        
    def _account2class(self, acc: List) -> List:
        return [self.mapping_account2class[c] for c in acc]