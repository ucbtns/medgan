"""
LOAD DATA from file.
"""

import os
import torch
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        #tuple_with_path = (original_tuple + (path,))
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
    
def load_data(opt):

    splits = ['train', 'test']
    drop_last_batch = {'train': True, 'test': False}
    shuffle = {'train': True, 'test': True}
    transform = transforms.Compose([transforms.Scale(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    
    dataset = {}
    dataset['train'] = ImageFolder(os.path.join(opt.dataroot, 'train'), transform)
    # Changed to include the path names
    dataset['test'] = ImageFolderWithPaths(os.path.join(opt.dataroot, 'test'), transform)

    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
    return dataloader

