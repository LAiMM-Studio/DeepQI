import numpy as np
import tensorflow.keras as K
import os
import matplotlib.pyplot as plt
import pandas as pd

from wigner_thermal import build_data
from data_merge import build_combined_data
from model import ResNet_base_decoder
from train import training, test_comparison
from cmap import rgb_cmap, rgb_cmap_inverse

build_data(dataset_filename = "npy_quantum", types = 'coherent', num = 20000, resolution = 0.0125, length = 4.5)
build_data(dataset_filename = "npy_quantum", types = 'cat', num = 20000, resolution = 0.0125, length = 4.5)

build_combined_data(dataset_filename = "npy_quantum", train_filename = "dataset_quantum", 
                    coherent_num = 16050, cat_num = 18050,
                    test_data_num = 100, split_num = 4, 
                    vmin = -0.45, vmax = 0.45)

training(dataset_filename = "dataset_quantum",  model_name = 'decoder_model_wigner_thermal',
          epochs = 100, batch_size = 32, data_patch = 4)

test_comparison(train_filename = "dataset_quantum",  model_name = 'decoder_model_wigner_thermal',
                  image_num = 5, vmin = -0.45, vmax = 0.45, point=721, step = 4.5)