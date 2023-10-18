# implemented by p0werHu
from .PEMSbase_dataset import PeMsDataset


class PeMs03Dataset(PeMsDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """

    def __init__(self, opt):
        """
        load data give options
        """
        dist_path = 'dataset/pems/PEMS03/adj_mx_03.pkl'
        data_path = 'dataset/pems/PEMS03/pems-03.h5'
        test_nodes_path = 'dataset/pems/PEMS03/test_nodes.npy'

        super().__init__(opt, dist_path, data_path, test_nodes_path)
