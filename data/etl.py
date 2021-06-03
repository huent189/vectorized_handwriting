import torch.nn as nn
import torch.utils.data as data
import pandas as pd
# import PIL.Image
import cv2
import os
import random
from argparse import ArgumentParser
import torchvision.transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .augmentation import Skeletonization, Binalization
from utils.vda.api import VDA, VDA_TPS
class CSVImageDatasets(data.Dataset):
    def __init__(self, csv_path, images_folder, config=None, is_val=False):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        if config.augment == 'binary':
            self.transform = A.Compose([
                            Binalization(),
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif config.augment == 'norm':
            self.transform = A.Compose([
                            Skeletonization(),
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif is_val or config.augment == 'none':
            self.transform = A.Compose([
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif config.augment == 'grid':
            self.transform = A.Compose(
                            [
                                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                A.GridDistortion(num_steps=8, p=config.augment_prob),
                                # A.ElasticTransform(alpha_affine=3, sigma=4, alpha=3),
                                A.Resize(64, 64),
                                ToTensorV2(),
                            ]
            )
        elif config.augment == 'elastic':
            self.transform = A.Compose(
                            [
                                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                # A.GridDistortion(num_steps=8),
                                # A.ElasticTransform(alpha_affine=3, sigma=4, alpha=3, p=config.augment_prob),
                                A.ElasticTransform(alpha_affine=10, sigma=15, alpha=8),
                                A.Resize(64, 64),
                                ToTensorV2(),
                            ]
            )
        elif config.augment == 'affine':
            self.transform = A.Compose(
                            [
                                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=config.augment_prob),
                                # A.GridDistortion(num_steps=8),
                                # A.ElasticTransform(alpha_affine=3, sigma=4, alpha=3),
                                A.Resize(64, 64),
                                ToTensorV2(),
                            ]
            )
        elif config.augment == 'album':
            self.transform = A.Compose(
                            [
                                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                # A.GridDistortion(num_steps=8),
                                A.ElasticTransform(alpha_affine=5, sigma=10, alpha=8),
                                A.Resize(64, 64),
                                ToTensorV2(),
                            ]
            )
        else:
            raise NotImplementedError()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.loc[index, "path"]
        label = self.df.loc[index, "label_idx"]
        hex_label = self.df.loc[index, "label"]
        image = cv2.imread(os.path.join(self.images_folder, filename), cv2.IMREAD_GRAYSCALE)
        # cv2.imwrite('/content/debug.png', image)
        assert image is not None, os.path.join(self.images_folder, filename)
        if self.transform is not None:
            image = self.transform(image=image)["image"].float()
            image = image / 255
        return image, label, filename, hex_label
    def get_num_classes(self):
        return max(self.df['label_idx']) + 1
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("data")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser
        parser.add_argument("--augment", type=str, default='none', choices=['none', 'album', 'norm', 'binary', 'grid', 'affine', 'elastic'], help="augmentation type")
        parser.add_argument("--augment_prob", type=float, default=0.5)
        return parser_out
class OfflineAugmentDataset(data.Dataset):
    def __init__(self, origin_path, images_folder, config=None, is_val=False):
        super().__init__()
        self.config = config
        self.df = pd.read_csv(origin_path)
        self.augment_df = pd.read_csv(config.augment_path)
        self.images_folder = images_folder
        if config.augment == 'binary':
            self.transform = A.Compose([
                            Binalization(),
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif config.augment == 'norm':
            self.transform = A.Compose([
                            Skeletonization(),
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif is_val or config.augment == 'none':
            self.transform = A.Compose([
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif config.augment == 'album':
            self.transform = A.Compose(
                            [
                                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                A.GridDistortion(num_steps=8),
                                A.Resize(64, 64),
                                ToTensorV2(),
                            ]
            )
        else:
            raise NotImplementedError()
    def __len__(self):
        return int(len(self.df) / self.config.augment_prob)
    def __getitem__(self, index):
        if index < len(self.df):
            filename = self.df.loc[index, "path"]
            label = self.df.loc[index, "label_idx"]
            hex_label = self.df.loc[index, "label"]
        else:
            index = random.randint(0, len(self.augment_df) - 1)
            filename = self.augment_df.loc[index, "path"]
            label = self.augment_df.loc[index, "label_idx"]
            hex_label = self.augment_df.loc[index, "label"]
        image = cv2.imread(os.path.join(self.images_folder, filename), cv2.IMREAD_GRAYSCALE)
        assert image is not None, os.path.join(self.images_folder, filename)
        if self.transform is not None:
            image = self.transform(image=image)["image"].float()
            image = image / 255
        return image, label, filename, hex_label
    def get_num_classes(self):
        return max(self.df['label_idx']) + 1
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("data")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser
        parser.add_argument("--augment", type=str, default='none', choices=['none', 'album', 'norm', 'binary'], help="augmentation type")
        parser.add_argument("--augment_prob", type=float, default=0.5)
        parser.add_argument("--augment_path")
        return parser_out
class OfflineVDADataset(data.Dataset):
    def __init__(self, origin_path, images_folder, config=None, is_val=False):
        super().__init__()
        self.config = config
        self.df = pd.read_csv(origin_path)
        # self.augment_df = pd.read_csv(config.augment_path)
        self.augment_root = config.augment_root
        self.augment_len = config.augment_len
        self.images_folder = images_folder
        if config.augment == 'binary':
            self.transform = A.Compose([
                            Binalization(),
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif config.augment == 'norm':
            self.transform = A.Compose([
                            Skeletonization(),
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif is_val or config.augment == 'none':
            self.transform = A.Compose([
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif config.augment == 'album':
            self.transform = A.Compose(
                            [
                                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                A.GridDistortion(num_steps=8),
                                A.Resize(64, 64),
                                ToTensorV2(),
                            ]
            )
        else:
            raise NotImplementedError()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.loc[index, "path"]
        print(filename)
        label = self.df.loc[index, "label_idx"]
        hex_label = self.df.loc[index, "label"]
        is_real = random.random() < self.config.augment_prob
        if not is_real:
            random_idx = random.randint(0, self.augment_len - 1)
            basename = os.path.split(filename)[-1].split('.')[0]
            filename = os.path.join(self.augment_root, f'{basename}_augment_{random_idx}.png')
        print(os.path.join(self.images_folder, filename))
        image = cv2.imread(os.path.join(self.images_folder, filename), cv2.IMREAD_GRAYSCALE)
        assert image is not None, os.path.join(self.images_folder, filename)
        if self.transform is not None:
            image = self.transform(image=image)["image"].float()
            image = image / 255
        return image, label, filename, hex_label
    def get_num_classes(self):
        return max(self.df['label_idx']) + 1
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("data")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser
        parser.add_argument("--augment", type=str, default='none', choices=['none', 'album', 'norm', 'binary'], help="augmentation type")
        parser.add_argument("--augment_prob", type=float, default=0.5)
        parser.add_argument("--augment_root")
        parser.add_argument("--augment_len", type=int)
        return parser_out
class OnlineVDADataset(data.Dataset):
    def __init__(self, origin_path, images_folder, config=None, is_val=False):
        super().__init__()
        self.config = config
        if config.use_tps:
            self.vda = VDA_TPS()
        else:
            self.vda = VDA()
            self.vda.update_config(config)
        self.df = pd.read_csv(origin_path)
        # self.augment_df = pd.read_csv(config.augment_path)
        # self.augment_root = config.augment_root
        # self.augment_len = config.augment_len
        self.images_folder = images_folder
        if config.augment == 'binary':
            self.transform = A.Compose([
                            Binalization(),
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif config.augment == 'norm':
            self.transform = A.Compose([
                            Skeletonization(),
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif is_val or config.augment == 'none':
            self.transform = A.Compose([
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        elif config.augment == 'album':
            self.transform = A.Compose(
                            [
                                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                A.GridDistortion(num_steps=8),
                                A.Resize(64, 64),
                                ToTensorV2(),
                            ]
            )
        else:
            raise NotImplementedError()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.loc[index, "path"]
        # print(filename)
        label = self.df.loc[index, "label_idx"]
        hex_label = self.df.loc[index, "label"]
        is_fake = random.random() < self.config.augment_prob
        if is_fake:
            # print('augment')
            image = self.vda.augment_one_image(filename)
        else:
            image = cv2.imread(os.path.join(self.images_folder, filename), cv2.IMREAD_GRAYSCALE)
        assert image is not None, os.path.join(self.images_folder, filename)
        if self.transform is not None:
            image = self.transform(image=image)["image"].float()
            image = image / 255
        return image, label, filename, hex_label
    def get_num_classes(self):
        return max(self.df['label_idx']) + 1
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("data")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser
        # parser.add_argument("--augment", type=str, default='none', choices=['none', 'album', 'norm', 'binary'], help="augmentation type")
        # parser.add_argument("--augment_prob", type=float, default=0.5)
        # parser.add_argument("--augment_root")
        parser.add_argument("--use_tps", action='store_true')
        # parser.add_argument("--augment_len", type=int)

        return parser_out
if __name__ == '__main__':
    dataset = CSVImageDatasets('/content/images/all/val.csv', '/content/images/all/', transform=[torchvision.transforms.Resize((64,64))])
    print(dataset.__getitem__(0))
    print(dataset.get_num_classes())