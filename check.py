#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import pickle

if __name__ == '__main__':
    print('cuda is available:', torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device, (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))
