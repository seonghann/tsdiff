import yaml
import torch
from GeoDiff.models.epsnet import get_model

with open("configs/ts_pretrain_exp1.yml", "r") as f:
    config = yaml.safe_load(f)

ckpt = torch.load(
    "logs/drugs_default_2022_10_12__21_47_06/checkpoints/3000000.pt", map_location="cpu"
)
