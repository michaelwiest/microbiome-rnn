import numpy as np
import random
from torch.autograd import Variable
import torch

def add_cuda_to_variable(data_nums, is_gpu, requires_grad=True):
    tensor = torch.FloatTensor(data_nums, requires_grad=requires_grad)
    if len(tensor.size()) == 1:
        tensor = tensor.unsqueeze_(0).unsqueeze(0).transpose(0, 2)
    else:
        pass
        # tensor = tensor.unsqueeze_(len(tensor.size()))
    if is_gpu:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
