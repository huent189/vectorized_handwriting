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
    parser.add_argument('--train_csv')
    parser.add_argument('--val_csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifyBase.add_argparse_args(parser)
    parser = CSVImageDatasets.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)
    train_dataset = CSVImageDatasets(args.train_csv, args.data_root, config=args)
    val_dataset = CSVImageDatasets(args.val_csv, args.data_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=4)
    args.n_classes = train_dataset.get_num_classes()
    model = LitClassifyBase(args)
    checkpoint_callback = ModelCheckpoint(
        monitor='accuracy',
        filename='etl-{epoch:02d}-{accuracy:.2f}',
        save_top_k=3,
        mode='max',
    )
    args.checkpoint_callback = checkpoint_callback
    print(train_dataset.__len__())
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    cli_main()
    