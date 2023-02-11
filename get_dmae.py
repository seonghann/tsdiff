import torch
import pickle
import glob

def dmae_fn(d):
    N = d.pos.shape[0]
    dm_gen = torch.cdist(d.pos_gen, d.pos_gen)
    dm_ref = torch.cdist(d.pos, d.pos)
    dmae = torch.abs(dm_gen - dm_ref).triu(1).sum()/((N-1) * N / 2)
    return dmae

def pmae_fn(d):
    N = d.pos.shape[0]
    dm_gen = torch.cdist(d.pos_gen, d.pos_gen)
    dm_ref = torch.cdist(d.pos, d.pos)
    pmae = (torch.abs(dm_gen - dm_ref)/dm_ref).triu(1).sum()/((N-1) * N / 2) * 100
    return pmae

#files = glob.glob("logs/train_schnet_e*/sample_ld_exp0208_seed2023/samples_all.pkl")
files = glob.glob("logs/job_coff10.0*/sample_ld_exp0208_seed2023/samples_all.pkl")
print(files)
for fn in files:
    with open(fn, "rb") as f:
        data = pickle.load(f)

    dmae = torch.Tensor([dmae_fn(d) for d in data])
    pmae = torch.Tensor([pmae_fn(d) for d in data])
    dmae = dmae.mean()
    pmae = pmae.mean()
    print(fn)
    print(dmae, pmae)
    print()
