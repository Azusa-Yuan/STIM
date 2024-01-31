import time
from typing import Dict

import setproctitle
import torch
from torch import nn
from torch.utils.data import DataLoader
from easytorch import Runner
from easytorch.utils import master_only
from easytorch.core.data_loader import build_data_loader


class BaseRunner(Runner):
    """
        An expanded easytorch runner for benchmarking time series models.
            - Support test loader and test process.
            - Support setup_graph for the models acting like tensorflow.
    """

    def __init__(self, cfg: dict):
        """Init

        Args:
            cfg (dict): all in one configurations
        """

        super().__init__(cfg)

        # validate every `val_interval` epoch
        self.val_interval = cfg["VAL"].get("INTERVAL", 1)
        self.val_start = cfg["VAL"].get("START", 1)
        self.val_best = False

        # declare data loader
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        # declare test parameters
        self.test_best = None
        self.test_start = None
        self.test_interval = None
        self.evaluate_on_gpu = None

        # fit higher easy-torch version
        if not hasattr(self,"to_running_device"):
            from easytorch.device import to_device
            self.to_running_device = to_device

        # set proctitle
        proctitle_name = "{0}({1})".format(cfg["MODEL"].get(
            "NAME", " "), cfg.get("DATASET_NAME", " "))
        setproctitle.setproctitle("{0}@BasicTS".format(proctitle_name))

    @staticmethod
    def define_model(cfg: Dict) -> nn.Module:
        return cfg["MODEL"]["ARCH"](**cfg.MODEL.PARAM)

    def build_train_data_loader(self, cfg: dict) -> DataLoader:
        """Support "setup_graph" for the models acting like tensorflow.

        Args:
            cfg (dict): all in one configurations

        Returns:
            DataLoader: train dataloader
        """

        train_data_loader = super().build_train_data_loader(cfg)
        if cfg["TRAIN"].get("SETUP_GRAPH", False):
            for data in train_data_loader:
                self.setup_graph(data)
                break
        return train_data_loader

    def setup_graph(self, data: torch.Tensor):
        """Setup all parameters and the computation graph.

        Args:
            data (torch.Tensor): data necessary for a forward pass
        """

        pass

    def init_training(self, cfg: dict):
        """Initialize training and support test dataloader.

        Args:
            cfg (dict): config
        """

        super().init_training(cfg)
        # init test
        if hasattr(cfg, "TEST"):
            self.init_test(cfg)

    @master_only
    def init_test(self, cfg: dict):
        """Initialize test.

        Args:
            cfg (dict): config
        """

        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_epoch_meter("test_time", "test", "{:.2f} (s)", plt=False)
        # test every `test_interval` epoch from START epoch
        self.test_interval = cfg["TEST"].get("INTERVAL", 1)
        self.test_start = cfg["TEST"].get("START", 1)
        # only test best val
        self.test_best = cfg["TEST"].get("BEST", False)
        self.evaluate_on_gpu = cfg["TEST"].get("USE_GPU", True)     # evaluate on gpu or cpu (gpu is faster but may cause OOM)


    def build_test_data_loader(self, cfg: dict) -> DataLoader:
        """Build val dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader```.

        Args:
            cfg (dict): config

        Returns:
            val data loader (DataLoader)
        """

        dataset = self.build_test_dataset(cfg)
        return build_data_loader(dataset, cfg["TEST"]["DATA"])

    @staticmethod
    def build_test_dataset(cfg: dict):
        """It can be implemented to a build dataset for test.

        Args:
            cfg (dict): config

        Returns:
            val dataset (Dataset)
        """

        raise NotImplementedError()

    # support test process
    def on_epoch_end(self, epoch: int):
        """Callback at the end of an epoch.

        Args:
            epoch (int): current epoch.
        """

        # print train meters
        self.print_epoch_meters("train")
        # tensorboard plt meters
        self.plt_epoch_meters("train", epoch)
        # validate
        if self.val_data_loader is not None and epoch % self.val_interval == 0 and epoch >= self.val_start:
            self.validate(train_epoch=epoch)
        # test
        if self.test_data_loader is not None and epoch % self.test_interval == 0 and epoch >= self.test_start:
            if self.test_best:
                if self.val_best:
                    self.test_process(train_epoch=epoch)
            else:
                self.test_process(train_epoch=epoch)
        # save model
        self.save_model(epoch)
        # reset meters
        self.reset_epoch_meters()

    @torch.no_grad()
    @master_only
    def test_process(self, cfg: dict = None, train_epoch: int = None):
        """The whole test process.

        Args:
            cfg (dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init test if not in training process
        if train_epoch is None:
            self.init_test(cfg)

        self.on_test_start()

        test_start_time = time.time()
        self.model.eval()

        # test
        self.test()

        test_end_time = time.time()
        self.update_epoch_meter("test_time", test_end_time - test_start_time)
        # print test meters
        self.print_epoch_meters("test")
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters("test", train_epoch // self.test_interval)

        self.on_test_end()

    @master_only
    def on_test_start(self):
        """Callback at the start of testing.
        """

        pass

    @master_only
    def on_test_end(self):
        """Callback at the end of testing.
        """

        pass

    def test(self, train_epoch: int = None):
        """It can be implemented to define testing details.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        raise NotImplementedError()
