import torch
from torch_geometric.data import Data, Batch

torch.manual_seed(1234)


def get_initial_guess(D):
    M = (D[:, 0].unsqueeze(1) ** 2 + D[0, :].unsqueeze(0) ** 2 - D**2) / 2
    u, s, _ = torch.linalg.svd(M)
    x_init = u @ torch.diag(torch.sqrt(s))
    x_init = x_init[:, :3]
    return x_init


def get_distance_matrix(x):
    D = (x.unsqueeze(0) - x.unsqueeze(1)).norm(dim=-1)
    # return torch.cdist(x,x)
    return D


def get_coord_from_distance(x, D, steps, lr=0.01, patience=1000, device="cpu"):
    D = D.to(device)
    x = x.clone()
    x.requires_grad = True
    opt = torch.optim.Adam(params=(x,), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", patience=patience
    )
    mask = torch.triu(torch.ones(D.shape, device=device), 1).bool()

    for i in range(steps):
        opt.zero_grad()
        D_gen = get_distance_matrix(x)
        logloss = 2 * torch.log(torch.abs(D[mask] - D_gen[mask])) - 4 * torch.log(
            torch.abs(D[mask])
        )
        loss = torch.exp(logloss).sum()
        print(loss)
        if torch.isnan(loss).item():
            raise ValueError("Nan detected during coordinate optimization")
        loss.backward()
        opt.step()
        scheduler.step(loss)

    out = x.detach()
    if device != "cpu":
        out = out.cpu()

    return out


def get_coord_from_distance_parallel(
    x_list, D_list, steps, lr=0.01, patience=1000, device="cuda"
):
    def _make_data(x, D):
        n = len(x)
        edge_index = []
        for i in range(1, n):
            for j in range(i):
                edge_index.append((i, j))
        edge_index = torch.LongTensor(edge_index).T
        D_target = D[edge_index[0], edge_index[1]]
        return Data(pos=x, edge_index=edge_index, target=D_target)

    batch = [_make_data(x, D) for x, D in zip(x_list, D_list)]
    batch = Batch.from_data_list(batch).to(device)
    pos = batch.pos
    e = batch.edge_index
    D_target = batch.target

    pos.requires_grad = True
    opt = torch.optim.Adam(params=(pos,), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", patience=patience
    )
    for i in range(steps):
        opt.zero_grad()
        D_gen = (pos[e[0]] - pos[e[1]]).norm(dim=-1)
        # logloss = 2 * torch.log(torch.abs(D_gen - D_target)) - 4 * torch.log(
        logloss = 2 * torch.log(torch.abs(D_gen - D_target) + 1e-12) - 4 * torch.log(
            torch.abs(D_target)
        )
        loss = torch.exp(logloss).sum()
        if torch.isnan(loss).item():
            raise ValueError("Nan detected during coordinate optimization")
            break

        loss.backward()
        # loss_.backward()
        opt.step()
        scheduler.step(loss)

    out = [data.pos.detach().to("cpu") for data in batch.to_data_list()]
    return out


def get_coord_from_ditance_old(D, steps, lr=0.01, patience=1000, device="cpu"):
    D = D.to(device)
    x = get_initial_guess(D)
    x.requires_grad = True
    opt = torch.optim.Adam(params=(x,), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", patience=patience
    )
    for i in range(steps):
        opt.zero_grad()
        D_gen = get_distance_matrix(x)
        loss = torch.triu(torch.abs(D - D_gen), 1).sum()
        loss.backward()
        opt.step()
        scheduler.step(loss)

    out = x.detach()
    if device != "cpu":
        out = out.cpu()

    return out


if __name__ == "__main__":
    x = torch.randn(10, 3)
    D = get_distance_matrix(x)
    x_gen = [get_coord_from_ditance(D, s) for s in [100, 500, 1000, 5000, 10000]]
    D_gen = [get_distance_matrix(x) for x in x_gen]

    for D_ in D_gen:
        print(torch.abs(D - D_).mean())
