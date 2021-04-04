import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import PIL.Image
import os
from argparse import ArgumentParser
import torchvision.transforms
class CSVImageDatasets(data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        if transform is None:
            self.transform = torchvision.transforms.ToTensor()
        else:
            self.transform = torchvision.transforms.Compose(transform + [torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.loc[index, "path"]
        label = self.df.loc[index, "label_idx"]
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def get_num_classes(self):
        return max(self.df['label_idx']) + 1
    # @staticmethod
    # def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
    #     if use_argument_group:
    #         parser = parent_parser.add_argument_group("data")
    #         parser_out = parent_parser
    #     else:
    #         parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #         parser_out = parser
    #     parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    #     parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    #     parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    #     return parser_out
if __name__ == '__main__':
    dataset = CSVImageDatasets('/content/images/all/val.csv', '/content/images/all/', transform=[torchvision.transforms.Resize((64,64))])
    print(dataset.__getitem__(0))
    print(dataset.get_num_classes())