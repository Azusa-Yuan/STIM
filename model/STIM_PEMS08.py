import os
import sys

from basicts.utils import load_adj, load_pkl

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.losses import masked_mae
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners.runner_zoo.simple_tsf_runner import SimpleTimeSeriesForecastingRunner
from model.model_arch.STIM.STIM_attention import STIM


CFG = EasyDict()


# ================= general ================= #
CFG.DESCRIPTION = "STIM model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS08"
CFG.DATASET_TYPE = "Traffic Flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

node_feats_full = load_pkl("datasets/{0}/data_in{1}_out{2}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN))["processed_data"]
train_index_list = load_pkl("datasets/{0}/index_in{1}_out{2}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN))["train"]
node_feats = node_feats_full[:train_index_list[-1][-1], ...]
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "original")
adj_mx = adj_mx[0]
node, _ = adj_mx.shape

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STIM"
CFG.MODEL.ARCH = STIM
CFG.MODEL.PARAM = {
    "num_nodes": node,
    "in_steps": 12,
    "out_steps": 12,
    "steps_per_day": 288,
    "input_dim": 3,
    "output_dim": 1,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "dow_embedding_dim": 24,
    "spatial_embedding_dim": 40,
    "adaptive_embedding_dim": 40,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.15,
    "node_feats": node_feats[-6000:],
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2] # traffic flow, time in day
CFG.MODEL.TARGET_FEATURES = [0]# traffic flow

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0015,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [15, 30, 40],
    "gamma": 0.1
}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False
