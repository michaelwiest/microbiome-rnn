import numpy as np
import random
from torch.autograd import Variable
import torch

def add_cuda_to_variable(data_nums, is_gpu):
    tensor = torch.FloatTensor(data_nums)
    if isinstance(data_nums, list):
        tensor = tensor.unsqueeze_(0)
    tensor = tensor.unsqueeze_(2)
    if is_gpu:
        return Variable(tensor.cuda())[:, :, 0]
    else:
        return Variable(tensor)[:, :, 0]
