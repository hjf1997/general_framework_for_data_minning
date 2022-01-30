# implemented by p0werHu
from data.base_dataset import BaseDataset

class DemoDataset(BaseDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults()
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """
        load data give options
        """
        pass

    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        """
        return {}

