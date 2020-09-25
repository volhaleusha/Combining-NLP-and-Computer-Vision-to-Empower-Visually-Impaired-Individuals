import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable


def load_image(image_path, transform=None):
    '''
    A helper function to load images from image path
    args: 
        image_path: folder with images (str)
        transform: Apply transform on image (None or function)
    return:
        image
    '''
    
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image


def create_masks(trg):
    """
    a function that creates mask for transformer model: need to prevent the first output predictions from being able to see later into the sentence (for more details refer to : https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)
    args:
        trg: captions without last word
    return:
        trg_mask
    
    """
    # for non empty targets unsqeeze -2 dimesion:
    trg_mask = (trg != 0 ).unsqueeze(-2)
     # get seq_len for matrix
    size = trg.size(1)
    #initialize np_mask as triangular matrix or zeros and ones
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).to(device)
    # apply np_mask on trg_mask
    trg_mask = trg_mask & np_mask

    return trg_mask

