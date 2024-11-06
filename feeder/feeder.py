# np
import numpy as np

# torch
import torch

class Feeder(torch.utils.data.Dataset):
    """ Feeder for image segmentation
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, H, W, C)
        label_path: the path to label
        mode: must be train or test
    """

    def __init__(self,
                 data_path,
                 label_path,
                 mode,
                #  transform,
                #  label_transform,
                 ):
        self.mode = mode
        self.data_path = data_path
        self.label_path = label_path
        # self.transform = transform
        # self.label_transform = label_transform

        self.load_data()

    def load_data(self):
        # load label
        self.label = np.load(self.label_path)

        # load data
        self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        data_numpy = self.data[index]
        label = self.label[index]

        # processing
        # TODO

        return data_numpy, label
