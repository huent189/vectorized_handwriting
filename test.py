from argparse import ArgumentParser
import pytorch_lightning as pl
from model.classification import LitClassifyBase
from data.etl import CSVImageDatasets
from torch.utils.data import DataLoader
import torchvision.transforms
from pytorch_lightning.callbacks import ModelCheckpoint

def cli_main():
    pl.seed_everything(1234)
    parser = ArgumentParser()
    parser.add_argument('--data_root')
    parser.add_argument('--cp_path')
    parser.add_argument('--test_csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifyBase.add_argparse_args(parser)
    parser = CSVImageDatasets.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)
    test_dataset = CSVImageDatasets(args.test_csv, args.data_root, config=args, is_val=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=4)
    args.n_classes = 956
    model = LitClassifyBase.load_from_checkpoint(args.cp_path, config=args)
    print(test_dataset.__len__())
    trainer = pl.Trainer.from_argparse_args(args)
    result = trainer.test(model, test_dataloaders=test_loader,)
    print(result)

if __name__ == '__main__':
    cli_main()
    