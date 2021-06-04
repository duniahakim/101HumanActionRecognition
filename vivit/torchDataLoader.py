import torch
import numpy as np
from utils import compressed_pickle, decompress_pickle
import torchvision.transforms as transforms

class TorchDataLoader(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, dim=(10, 240, 320), n_channels=3,
                 n_classes=101):
        'Initialization'
        self.dim = dim
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.use_pretrained = use_pretrained
        # self.model = ResNet50(
        #                         include_top=False,
        #                         weights="imagenet",
        #                         input_tensor=None,
        #                         input_shape=(240, 320, 3),
        #                         pooling=None
        #                     )
        self.indexes = np.arange(len(self.list_IDs))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return len(self.list_IDs)

    def __getitem__(self, index):
       
        ID = self.list_IDs[index]
        X = np.array(decompress_pickle('../input/' + ID + '.pickle.pbz2'))
        y = self.labels[ID]
        return X, y