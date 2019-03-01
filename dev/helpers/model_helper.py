import numpy as np
import random
from torch.autograd import Variable
import torch

def add_cuda_to_variable(data_nums, is_gpu, requires_grad=True):
    '''
    Function for checking whether or not a GPU is available and
    making the var a cuda variable if so.
    '''
    tensor = torch.FloatTensor(data_nums)
    if len(tensor.size()) == 1:
        tensor = tensor.unsqueeze_(0).unsqueeze(0).transpose(0, 2)
    if is_gpu:
        return Variable(tensor.cuda(), requires_grad=requires_grad)
    else:
        return Variable(tensor, requires_grad=requires_grad)
