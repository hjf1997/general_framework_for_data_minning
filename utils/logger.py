import os
import time
import neptune.new as neptune
from neptune.new.types import File


class Logger():
    """This class includes several functions that can display/save image data, loss values and print/save logging information.
    It depends on the online experiment tracking platform neptune.ai (https://neptune.ai/)
    """

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.metrics_name = os.path.join(opt.checkpoints_dir, opt.name, 'matrices.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        with open(self.metrics_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Matrices (%s) ================\n' % now)

        # neptune experiment tracking
        if opt.isTrain and opt.enable_neptune:
            try:
                self.neptune_run = neptune.init(project=opt.neptune_project,
                                   api_token=opt.neptune_token,
                                   source_files=['*.py'])
            except Exception as e:
                print(e)
                opt.enable_neptune = False
            self.neptune_options(opt)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.5f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_metrics(self, epoch, iters, metrics, t_val):
        """print current losses on console; also save the losses to the disk
        """
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iters, t_val)
        for k, v in metrics.items():
            message += '%s: %.5f ' % (k, v)

        print(message)  # print the message
        with open(self.matrices_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def neptune_options(self, opt):
        """
        print configurations to neptune
        :return:
        """
        config = {}
        for k, v in sorted(vars(opt).items()):
            config[k] = v
        self.neptune_run['configurations'] = config

    def neptune_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses to neptune;
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        for k, v in losses.items():
            self.neptune_run['train/'+k].log('%.4f' % v)
        self.neptune_run['train/computation time'].log('%.4f' % t_comp)
        self.neptune_run['train/data loading time'].log('%.4f' % t_data)

    def neptune_current_metrics(self, epoch, iters, metrics, t_val):
        """
        print metrics to neptune
        :param epoch:
        :param iters:
        :param metrics:
        :param t_val:
        :return:
        """
        for k, v in metrics.items():
            self.neptune_run['train/validation'+k].log('%.4f' % v)
        self.neptune_run['train/validation/computation time'].log('%.4f' % t_val)

    def neptune_networks(self, model):
        """
        print the total number of parameter in the network to neptune
        :param model:
        :return:
        """
        for name in model.model_names:
            if isinstance(name, str):
                net = eval('model.net'+name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
            self.neptune_run['model/parameters/'+name] = num_params

    def neptune_visuals(self, visual):
        """
        upload
        :param visual:
        :return:
        """
        for k, v in visual.items():
            self.neptune_run['visualizations/' + k].log(File.as_image(v))
