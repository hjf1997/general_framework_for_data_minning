# implemented by p0werHu
from . import init_net, BaseModel
import torch.nn as nn
import torch


class DemoModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # modify options for the model
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []

        # specify the any image data you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = []

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = []

        # define networks. The model variable should begin with 'self.net'
        self.net = None
        self.net = init_net(self.net, opt.init_type, opt.init_gain, opt.gpu_ids)  # initialize parameters, move to cuda if applicable

        # define loss functions
        if self.isTrain:
            self.criterion = nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """
        parse input for one epoch. data should be stored as self.xxx which would be adopted in self.forward().
        :param input: dict
        :return: None
        """
        pass

    def forward(self):
        pass

    def backward(self):
        # loss values expected to be displayed should begin with 'self.loss'
        self.loss_xxx = None

        self.loss_xxx.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.net, True)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

