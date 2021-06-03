from argparse import ArgumentParser
import pytorch_lightning as pl
from model.classification import LitClassifyBase
from data.etl import CSVImageDatasets, OfflineVDADataset, OnlineVDADataset
from torch.utils.data import DataLoader
import torchvision.transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
def cli_main():
    pl.seed_everything(42)
    parser = ArgumentParser()
    parser.add_argument('--data_root')
    parser.add_argument('--train_csv')
    parser.add_argument('--val_csv')
    parser.add_argument('--test_csv')
    parser.add_argument('--ver_name')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--method_augment', type=str, choices=['on', 'off', None])
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifyBase.add_argparse_args(parser)
    parser = CSVImageDatasets.add_argparse_args(parser)
    parser = OnlineVDADataset.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)
    print(args.method_augment)
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
    # model = LitClassifyBase(args)
    model = LitClassifyBase(args)

    # from tqdm import tqdm
    # import os
    # import cv2
    # for i, data in tqdm(enumerate(train_dataset)):
    #     img = data[0] * 255.0
    #     img = img.numpy()[0]
    #     name = f'/content/log/imgs/{i}.png'
    #     cv2.imwrite(name, img)

    checkpoint_callback = ModelCheckpoint(
        monitor='accuracy',
        filename='etl-{epoch:02d}-{accuracy:.4f}',
        save_top_k=3,
        mode='max',
        save_last = True
    )
    early_stop_callback = EarlyStopping(
            monitor='accuracy',
            min_delta=0.00,
            patience=15,
            verbose=False,
            mode='max'
            )
    args.checkpoint_callback = checkpoint_callback
    print(train_dataset.__len__())
    trainer = pl.Trainer.from_argparse_args(args, callbacks = [early_stop_callback])
    trainer.logger =  pl.loggers.TensorBoardLogger(
                save_dir = args.default_root_dir,
                version=args.ver_name,
                name = 'same_writer'
            )
    trainer.fit(model, train_loader, val_loader)
    test_dataset = CSVImageDatasets(args.test_csv, args.data_root, config=args, is_val=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=4)
    result = trainer.test(test_dataloaders=test_loader)
    print(result)

if __name__ == '__main__':
    cli_main()
    