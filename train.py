from argparse import ArgumentParser
import pytorch_lightning as pl
from model.classification import LitClassifyBase
from data.etl import CSVImageDatasets, OfflineVDADataset, OnlineVDADataset
from torch.utils.data import DataLoader
import torchvision.transforms
from pytorch_lightning.callbacks import ModelCheckpoint

def cli_main():
    pl.seed_everything(46)
    parser = ArgumentParser()
    parser.add_argument('--data_root')
    parser.add_argument('--train_csv')
    parser.add_argument('--val_csv')
    parser.add_argument('--test_csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--method_augment', type=str, choices=['on', 'off', None])
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifyBase.add_argparse_args(parser)
    parser = OfflineVDADataset.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)
    if args.method_augment == 'off':
        train_dataset = OfflineVDADataset(args.train_csv, args.data_root, config=args)
    elif args.method_augment == 'on':
        train_dataset = OnlineVDADataset(args.train_csv, args.data_root, config=args)
    else:
        train_dataset = CSVImageDatasets(args.train_csv, args.data_root, config=args)
    val_dataset = CSVImageDatasets(args.val_csv, args.data_root, config=args, is_val=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=4, shuffle=True)
    args.n_classes = train_dataset.get_num_classes()
    model = LitClassifyBase(args)
    checkpoint_callback = ModelCheckpoint(
        monitor='accuracy',
        filename='etl-{epoch:02d}-{accuracy:.4f}',
        save_top_k=3,
        mode='max',
        save_last = True
    )
    args.checkpoint_callback = checkpoint_callback
    print(train_dataset.__len__())
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)
    test_dataset = CSVImageDatasets(args.test_csv, args.data_root, config=args, is_val=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=4)
    result = trainer.test(test_dataloaders=test_loader)
    print(result)

if __name__ == '__main__':
    cli_main()
    