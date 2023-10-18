# implemented by p0werHu
import os
import time
from options.train_options import TrainOptions
from options.val_options import Valptions
from data import create_dataset
from models import create_model
from utils.logger import Logger
import subprocess

if __name__ == '__main__':
    opt, model_config = TrainOptions().parse()   # get training options
    visualizer = Logger(opt)  # create a visualizer that display/save and plots
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of samples in the dataset.
    print('The number of training samples = %d' % dataset_size)

    if opt.enable_val:
        val_opt, _ = Valptions().parse()  # get validation options
        val_dataset = create_dataset(val_opt)  # create a validation dataset given opt.dataset_mode and other options
        dataset_size = len(val_dataset)  # get the number of samples in the dataset.
        print('The number of validation samples = %d' % dataset_size)

    model = create_model(opt, model_config)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    best_metric = float('inf')  # best metric
    n_target_start = opt.num_train_target
    early_stop_trigger = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time, iter_data_time = time.time()
        model.train()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            iter_start_time = time.time()  # timer for computation per iteration
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            total_iters += 1
            if total_iters % opt.print_freq == 0:   # display
                losses = model.get_current_losses()
                t_comp = time.time() - iter_start_time  # time for one iter optimization
                t_data = iter_start_time - iter_data_time  # time for one iter data loading
                visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)
            iter_data_time = time.time()
        epoch_end_time = time.time()

        if opt.enable_val and epoch % val_opt.eval_epoch_freq == 0:
            model.eval()
            val_start_time = time.time()
            for i, data in enumerate(val_dataset):  # inner loop within one epoch
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.test()
                model.cache_results()  # store current batch results
            t_val = time.time() - val_start_time
            model.compute_metrics()
            metrics = model.get_current_metrics()
            visualizer.print_current_metrics(epoch, total_iters, metrics, t_val)

            if opt.save_best and best_metric > metrics['MAE']:
                print('saving the best model at the end of epoch %d, iters %d' % (epoch, total_iters))
                best_metric = metrics['MAE']
                model.save_networks('best')
                early_stop_trigger = 0
            else:
                early_stop_trigger += val_opt.eval_epoch_freq
            model.clear_cache()

            # check early stopping
            early_stopping_threshold = 40  # todo: make it a parameter
            if early_stop_trigger >= early_stopping_threshold:
                print('Trigger early stopping!')
                break

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, epoch_end_time - epoch_start_time))
        new_lr = model.update_learning_rate()  # update learning rates in the beginning of every epoc

    print('Run evaluation')
    with open(os.path.join(model.save_dir,'run_test.sh'), 'w') as f:
        cmd = 'python test.py --model {} --dataset_mode {} ' \
                  ' --gpu_ids {} --config {} --t_len {} --file_time {} --epoch best'.format(
            opt.model,
            opt.dataset_mode,
            opt.gpu_ids[0]+1,
            opt.config,
            opt.t_len,
            opt.file_time)
        f.write(cmd)

    os.system('chmod u+x '+ os.path.join(model.save_dir,'run_test.sh'))
    subprocess.Popen(os.path.join(model.save_dir, 'run_test.sh'), shell=True)
