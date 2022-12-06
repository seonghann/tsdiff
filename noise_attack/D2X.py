import torch

torch.manual_seed(1234)


def get_initial_guess(D):
    M = (D[:,0].unsqueeze(1)**2 + D[0,:].unsqueeze(0)**2 - D**2)/2
    u,s,_ = torch.linalg.svd(M)
    x_init = u@torch.diag(torch.sqrt(s))
    x_init = x_init[:,:3]
    return x_init

def get_distance_matrix(x):
    D = (x.unsqueeze(0) - x.unsqueeze(1)).norm(dim=-1)
    return D

def get_coord_from_ditance(x, D, steps, lr=0.01, patience=1000, device="cpu"):
    D = D.to(device)
    #x = get_initial_guess(D)
    x = x.clone()
    x.requires_grad = True
    opt = torch.optim.Adam(params = (x,), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=patience)
    mask = torch.triu(torch.ones(D.shape, device=device),1).bool()

    for i in range(steps):
        opt.zero_grad()
        D_gen = get_distance_matrix(x)
        logloss = 2*torch.log(torch.abs(D[mask] - D_gen[mask])) - 4*torch.log(torch.abs(D[mask]))
        loss = torch.exp(logloss).sum()
        #loss = torch.triu(torch.abs(D - D_gen),1).sum()
        loss.backward()
        opt.step()
        scheduler.step(loss)
    
    out = x.detach()
    if device != "cpu":
        out = out.cpu()

    return out

def get_coord_from_ditance_old(D, steps, lr=0.01, patience=1000, device="cpu"):
    D = D.to(device)
    x = get_initial_guess(D)
    x.requires_grad = True
    opt = torch.optim.Adam(params = (x,), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=patience)
    for i in range(steps):
        opt.zero_grad()
        D_gen = get_distance_matrix(x)
        loss = torch.triu(torch.abs(D - D_gen),1).sum()
        loss.backward()
        opt.step()
        scheduler.step(loss)
    
    out = x.detach()
    if device != "cpu":
        out = out.cpu()

    return out

if __name__ == "__main__":
    x = torch.randn(10,3)
    D = get_distance_matrix(x)
    x_gen = [get_coord_from_ditance(D, s) for s in [100, 500, 1000, 5000, 10000]]
    D_gen = [get_distance_matrix(x) for x in x_gen]

    for D_ in D_gen:
        print(torch.abs(D - D_).mean())
