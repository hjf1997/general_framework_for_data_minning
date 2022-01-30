#  implemented by p0werHu
# time: 5/6/2021

from .base_options import BaseOptions


class Valptions():
    """This class includes validation options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        self.isTrain = False

        parser = BaseOptions.initialize(self, parser)
        # loading parameters
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        parser.set_defaults(serial_batches=True)

        return parser
