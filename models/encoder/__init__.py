from .schnet import SchNetEncoder
from .gin import GINEncoder
from .edge import *
from .coarse import *
from .comenet import ComENetEncoder
from .dimenetpp import DimeNetPPEncoder
from .egnn import EGNNMixed2DEncoder


EncoderDict = {
    "dimenetpp": DimeNetPPEncoder,
    "egnn": EGNNMixed2DEncoder,
    "schnet": SchNetEncoder,
    "gin": GINEncoder,
    "comenet": ComENetEncoder
}


def load_encoder(config, encoder_type="global_encoder"):
    cfg = config.get(encoder_type)
    encoder = EncoderDict[cfg.name].from_config(cfg)
    return encoder
