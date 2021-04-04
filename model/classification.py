import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from backbone.classification import ConvNet
from pytorch_lightning.metrics.classification import Accuracy
from argparse import ArgumentParser
import torch.optim
class LitClassifyBase(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self._get_backbone()
        self.acc = Accuracy()
        self.metrics = {'accuracy' : self.acc}
    def configure_optimizers(self):
        return torch.optim.Adam(self.backbone.parameters(), lr = self.config.lr, betas=[self.config.b1, self.config.b2])
    def _get_backbone(self):
        return ConvNet(self.config.n_classes)
    def _cal_loss(self, pred, y):
        return F.cross_entropy(pred, y)
    def training_step(self, batch, batch_idx):
        assert self.backbone.training
        x, y = batch
        pred = self.backbone(x)
        loss = self._cal_loss(pred, y)
        self.log('train_loss', loss, on_epoch=True,on_step=True)
        return loss
    def validation_step(self, batch, batch_idx):
        assert not self.backbone.training
        x, y = batch
        output = self.backbone(x)
        output = torch.argmax(output, dim=-1).to(x.device)
        # print(output.device, x.device, y.device)
        for m in self.metrics:
            self.log(m, self.metrics[m](output, y), on_epoch=True,on_step=False)
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("model")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser
        parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
        return parser_out