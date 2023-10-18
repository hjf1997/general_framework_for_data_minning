# implemented by p0werHu
from .PEMSbase_dataset import PeMsDataset

class PeMs04Dataset(PeMsDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """

    def __init__(self, opt):
        """
        load data give options
        """
        dist_path = 'dataset/pems/PEMS04/adj_mx_04.pkl'
        data_path = 'dataset/pems/PEMS04/pems-04.h5'
        test_nodes_path = 'dataset/pems/PEMS04/test_nodes.npy'

        super().__init__(opt, dist_path, data_path, test_nodes_path)
