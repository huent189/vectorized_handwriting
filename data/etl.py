import torch.nn as nn
import torch.utils.data as data
import pandas as pd
# import PIL.Image
import cv2
import os
from argparse import ArgumentParser
import torchvision.transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
class CSVImageDatasets(data.Dataset):
    def __init__(self, csv_path, images_folder, config=None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        if config is None or config.augment == 'none':
            self.transform = A.Compose([
                            A.Resize(64, 64),
                            ToTensorV2(),
                            ])
        else:
            self.transform = A.Compose(
                            [
                                A.SmallestMaxSize(max_size=64),
                                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                A.GridDistortion(num_steps=8),
                                ToTensorV2(),
                            ]
            )
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.loc[index, "path"]
        label = self.df.loc[index, "label_idx"]
        image = cv2.imread(os.path.join(self.images_folder, filename), cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            image = self.transform(image=image)["image"].float()
        return image, label
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
        parser.add_argument("--augment", type=str, default='none', choices=['none', 'album'], help="augmentation type")
        return parser_out
if __name__ == '__main__':
    dataset = CSVImageDatasets('/content/images/all/val.csv', '/content/images/all/', transform=[torchvision.transforms.Resize((64,64))])
    print(dataset.__getitem__(0))
    print(dataset.get_num_classes())