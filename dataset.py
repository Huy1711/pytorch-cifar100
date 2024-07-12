""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import glob
from PIL import Image

# from skimage import io
# import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset


class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image


class ETL952Dataset(Dataset):
    """ETL952 dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(
            self, 
            path, 
            vocab, 
            cangjie_dict, 
            transform=None
        ):
        self.data = glob.glob(os.path.join(path, "**/*.png"))
        self.vocab = vocab
        self.cangjie_dict = cangjie_dict
        
        # if transform is given, we transoform data using
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # the label is the name of the parent folder
        label = int(self.data[index].split("/")[-2])
        
        tokens = self.cangjie_dict[label]
        tokens = [self.vocab.index(token) for token in tokens]
        cangjie_label = torch.tensor(tokens, dtype=torch.long)
        
        image = Image.open(self.data[index])

        if self.transform:
            image = self.transform(image)
        return image, label, cangjie_label


class ETL952EvalDataset(Dataset):
    """ETL952 dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        self.data = glob.glob(os.path.join(path, "**/*.png"))
        # if transform is given, we transoform data using
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # the label is the name of the parent folder
        label = int(self.data[index].split("/")[-2])
        image = Image.open(self.data[index])

        if self.transform:
            image = self.transform(image)
        return image, label