# Implemented by p0werHu
# Time 13/06/2021
from scipy.io import loadmat

from data.base_dataset import BaseDataset
import os
import numpy as np
import datetime
import pickle
import pandas as pd


class MetrlaDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(y_dim=1, covariate_dim=1, spatial_dim=64)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        dist_path = 'dataset/metr/adj_mx_la.pkl'
        data_path = 'dataset/metr/metr-la.h5'
        test_nodes_path = 'dataset/metr/test_nodes.npy'
        self.time_division = {
            'train': [0.0, 0.7],
            'val': [0.7, 0.8],
            'test': [0.8, 1.0]
        }

        self.raw_data = self.load_feature(data_path, self.time_division[opt.phase])
        self.A = self.load_adj(dist_path)

        # divide data into train, val, test
        ## get division index
        self.test_node_index = self.get_node_division(test_nodes_path, num_nodes=self.raw_data['pred'].shape[0])
        self.train_node_index = np.setdiff1d(np.arange(self.raw_data['pred'].shape[0]), self.test_node_index)

        # data format check
        self._data_format_check()

    def load_feature(self, data_path, time_division, add_time_in_day=True, add_day_in_week=False):
        # load data

        df = pd.read_hdf(data_path)
        num_samples, num_nodes = df.shape
        X = np.expand_dims(df.values, axis=-1).transpose((1, 0, 2))
        feature_list = []
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list.append(time_in_day)
        if add_day_in_week:
            dow = df.index.dayofweek
            dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list.append(dow_tiled)
        feat = np.concatenate(feature_list, axis=-1).transpose((1, 0, 2))
        missing_index = np.zeros(X.shape)

        # time
        start_time = datetime.datetime.strptime('2012-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        time_list = [np.datetime64(start_time + t * datetime.timedelta(minutes=5)) for t in range(X.shape[1])]
        time_list = np.array(time_list)
        time_list = ((time_list - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')).astype(np.int64)

        # data normalization:
        self.add_norm_info(np.mean(X), np.std(X))
        X = (X - self.opt.mean) / self.opt.scale

        # division
        data_length = X.shape[1]
        start_index, end_index = int(time_division[0] * data_length), int(time_division[1] * data_length)
        X = X[:, start_index:end_index]
        missing_index = missing_index[:, start_index:end_index]
        feat = feat[:, start_index:end_index]
        time_list = time_list[start_index:end_index]

        data = {
            'time': time_list,
            'pred': X,
            'missing': missing_index,
            'feat': feat
        }
        return data

    def load_adj(self, pkl_filename):
        sensor_ids, sensor_id_to_ind, adj_mx = self.load_pickle(pkl_filename)
        return adj_mx

    def load_pickle(self, pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', pickle_file, ':', e)
            raise
        return pickle_data
