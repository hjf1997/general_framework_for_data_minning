# implemented by p0werHu

import time
from options.train_options import TrainOptions
from options.val_options import Valptions
from data import create_dataset
from models import create_model
from utils.logger import Logger


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    visualizer = Logger(opt)  # create a visualizer that display/save and plots
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of samples in the dataset.
    print('The number of training samples = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    if opt.val:
        val_opt = Valptions().parse()  # get validation options
        val_dataset = create_dataset(opt)  # create a validation dataset given opt.dataset_mode and other options
        dataset_size = len(val_dataset)  # get the number of samples in the dataset.
        print('The number of training samples = %d' % dataset_size)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        model.train()
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        display_iter = 0                # the number of data that been trained currently.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            epoch_iter += 1
            display_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:   # display images on visdom and save images to
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if opt.val and epoch and epoch % opt.eval_epoch_freq == 0:
            model.eval()
            val_start_time = time.time()
            for i, data in enumerate(val_dataset):  # inner loop within one epoch
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.compute_metrics()
                model.gather_metrcis()  # store the current batch metrics
            t_val = time.time() - val_start_time
            metrics = model.get_current_metrics()
            visualizer.print_current_metrics(epoch, total_iters, metrics, t_val)

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        new_lr = model.update_learning_rate()  # update learning rates in the beginning of every epoc