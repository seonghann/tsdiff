from GeoDiff.utils.activation_functions import swish
import torch.nn as nn
from GeoDiff.utils.datasets import generate_ts_data2

def activation_loader(name):
    if name == "swish":
        return swish()
    
    else:
        return getattr(nn, name)()

