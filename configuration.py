import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import pyworld
import librosa
import time
import matplotlib.pyplot as plt

from preprocess import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms as T
import random
from PIL import Image
import torch
import os
import random



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)

model_name = "model_lambda70_f1f4m1m4"

model_dir = "./model/" + model_name

data_dir =  "./vcc2018"
voice_dir_list = ["VCC2SF4", "VCC2SF1", "VCC2SM4", "VCC2SM1"]
output_dir = "./converted_voices/test/" + model_name + "_training_progress"
figure_dir = "./figure/" + model_name


lambda_p = 70
lambda_s = 70
nb_label = len(voice_dir_list)

num_epochs = 25
batch_size = 5
learning_rate =1e-3
learning_rate_ = 1e-4
learning_rate__ = 1e-5
learning_rate___ = 1e-6
sampling_rate = 22050
num_envelope  = 36
num_mcep = 36
frame_period = 5.0
n_frames = 1024

lambda_cls=1
lambda_rec=10
lambda_gp=10
