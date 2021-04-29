import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from backbone.classification import ConvNet
from pytorch_lightning.metrics.classification import Accuracy
from argparse import ArgumentParser
import torch.optim
from functools import reduce
import operator
import os
import csv
import torchvision
from utils.tboard import plot_classes_preds
class LitClassifyBase(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams = config
        self.backbone = self._get_backbone()
        self.acc = Accuracy()
        self.metrics = {'accuracy' : self.acc}
    def configure_optimizers(self):
        return torch.optim.Adam(self.backbone.parameters(), lr = self.hparams.lr, betas=[self.hparams.b1, self.hparams.b2])
    def _get_backbone(self):
        return ConvNet(self.hparams.n_classes)
    def _cal_loss(self, pred, y):
        return F.cross_entropy(pred, y)
    def training_step(self, batch, batch_idx):
        assert self.backbone.training
        x, y,_,_ = batch
        pred = self.backbone(x)
        loss = self._cal_loss(pred, y)
        self.log('train_loss', loss, on_epoch=True,on_step=True)
        if batch_idx == 0:
            self.logger.experiment.add_figure('training predictions vs. actuals', plot_classes_preds(pred, x, y), self.current_epoch)
        return loss
    def validation_step(self, batch, batch_idx):
        assert not self.backbone.training
        x, y,_,_ = batch
        preds = self.backbone(x)
        output = torch.argmax(preds, dim=-1).to(x.device)
        # print(output.device, x.device, y.device)

        for m in self.metrics:
            self.log(m, self.metrics[m](output, y), on_epoch=True,on_step=False)
        if batch_idx == 0:
            self.logger.experiment.add_figure('predictions vs. actuals', plot_classes_preds(preds, x, y), self.current_epoch)
    def test_step(self, batch, batch_idx):
        x, y, paths, hex_labels = batch
        y_hat = self.backbone(x)
        output = torch.argmax(y_hat, dim=-1).to(x.device)
        for m in self.metrics:
            self.log(f'test_{m}', self.metrics[m](output, y), on_epoch=True,on_step=False) 
        wrong_predict =  (output != y).nonzero(as_tuple=True)[0].tolist()
        return [[paths[i], output[i].item(), y[i].item(), hex_labels[i]] for i in wrong_predict]
    def test_epoch_end(self, outputs):
        rows = reduce(operator.concat,outputs)
        with open(os.path.join(self.logger.experiment.log_dir, self.hparams.bad_predict), 'a') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['path', 'pred', 'gt', 'gt_hex'])
            csv_writer.writerows(rows)
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
        parser.add_argument("--bad_predict", type=str, default='bad_predict.csv')
        return parser_out