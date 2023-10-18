# implemented by p0werHu
import datetime

from data.base_dataset import BaseDataset
from .PEMSbase_dataset import PeMsDataset


class PeMs08Dataset(PeMsDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """

    def __init__(self, opt):
        """
        load data give options
        """
        dist_path = 'dataset/pems/PEMS08/adj_mx_08.pkl'
        data_path = 'dataset/pems/PEMS08/pems-08.h5'
        test_nodes_path = 'dataset/pems/PEMS08/test_nodes.npy'

        super().__init__(opt, dist_path, data_path, test_nodes_path)
