import torch
import torch_geometric as tg
from dimenetpp import DimeNetPP

if __name__ == "__main__":
    torch.manual_seed(0)
    
    n_atomtypes = 10

    zs = [torch.randint(0, n_atomtypes, (4, )), torch.randint(0, n_atomtypes, (5, ))]
    poss =[torch.randn(4, 3), torch.randn(5, 3)]
    for pos in poss:
        dm = torch.cdist(pos, pos)
        print(dm)

    data = [tg.data.Data(z=z, pos=pos) for z, pos in zip(zs, poss)]
    batch = tg.data.Batch.from_data_list(data)

    model = DimeNetPP(cutoff=3.0, num_layers=2, basis_emb_size=8, out_emb_channels=256,
                      num_spherical=7, num_radial=5, )
    print(model(batch))
