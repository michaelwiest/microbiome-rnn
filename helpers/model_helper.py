import numpy as np
import random
from torch.autograd import Variable
import torch

def add_cuda_to_variable(data_nums, is_gpu):
    tensor = torch.FloatTensor(data_nums)
    if len(tensor.size()) == 1:
        tensor = tensor.unsqueeze_(0).unsqueeze(0).transpose(0, 2)
    else:
        tensor = tensor.unsqueeze_(len(tensor.size()))
    if is_gpu:
        return Variable(tensor.cuda())[:, :, 0]
    else:
        return Variable(tensor)[:, :, 0]
