# implemented by p0werHu
# time 9/6/2021

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.logger import Logger
import numpy as np
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of samples in the dataset.
    print('The number of training samples = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Logger(opt)  # create a visualizer that display/save and plots
    total_iters = 0                # the total number of training iterations

    model.eval()

    visualizer.print_current_metrics(int(opt.epoch))
