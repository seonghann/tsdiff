import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader

#from logs.models.epsnet import get_model
from models.epsnet import get_model
from utils.datasets import ConformationDataset, TSDataset
from utils.transforms import CountNodesPerGraph
from utils.misc import (
    seed_all,
    get_new_log_dir,
    get_logger,
    get_checkpoint_path,
    inf_iterator,
)
from utils.common import get_optimizer, get_scheduler

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_iter", type=int, default=None)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--pretrain", type=str, default="")
    parser.add_argument("--project", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--fn", type=str, default=None)
    args = parser.parse_args()
    torch.randn(1).to(args.device)

    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, "*.yml"))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[
        : os.path.basename(config_path).rfind(".")
    ]
    seed_all(config.train.seed)

    if args.tag is None:
        args.tag = args.name

    # Logging
    if resume:
        log_dir = get_new_log_dir(
            args.logdir, prefix=config_name, tag=f"{args.tag}_resume", fn=args.fn
        )
        os.symlink(
            os.path.realpath(resume_from),
            os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))),
        )
    else:
        #log_dir = get_new_log_dir(
        #    args.logdir, prefix=config_name, tag=f"{args.tag}", fn=args.fn
        #)
        #shutil.copytree("./models", os.path.join(log_dir, "models"))
        log_dir = None

    #ckpt_dir = os.path.join(log_dir, "checkpoints")
    #os.makedirs(ckpt_dir, exist_ok=True)
    #logger = get_logger("train", log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    #logger.info(args)
    #logger.info(config)

    use_wandb = False
    if args.name and args.project:
        use_wandb = True
        wandb.init(project=args.project, name=args.name)
        wandb.config = config

    #shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    #logger.info("Loading datasets...")
    transforms = CountNodesPerGraph()
    if hasattr(config.model, "TS") and config.model.TS:
        train_set = TSDataset(config.dataset.train, transform=transforms)
        val_set = TSDataset(config.dataset.val, transform=transforms)
    else:
        train_set = ConformationDataset(config.dataset.train, transform=transforms)
        val_set = ConformationDataset(config.dataset.val, transform=transforms)
    train_iterator = inf_iterator(
        DataLoader(train_set, config.train.batch_size, shuffle=True)
    )
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)

    # Model
    #logger.info("Building model...")
    model = get_model(config.model).to(args.device)
    # Optimizer
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    start_iter = 1

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(
            os.path.join(resume_from, "checkpoints"), it=args.resume_iter
        )
        #logger.info("Resuming from: %s" % ckpt_path)
        #logger.info("Iteration: %d" % start_iter)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    if args.pretrain:
        #logger.info(f"pretraining model checkpoint load : {args.pretrain}")
        print(args.pretrain)
        ckpt = torch.load(args.pretrain, map_location=args.device)
        model.load_state_dict(ckpt["model"], strict=False)

    def check_contribution(it, time_t0, time_t1):
        sum_n = 0
        sum_losses = [0 for i in range(7)]
        sum_biases = [0 for i in range(7)]
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_loader):
                batch = batch.to(args.device)
                losses, biases = model._check_contribution(
                    atom_type=batch.atom_type,
                    r_feat=batch.r_feat,
                    p_feat=batch.p_feat,
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    t0=time_t0,
                    t1=time_t1,
                )
                
                for i, (loss, bias) in enumerate(zip(losses, biases)):
                    sum_losses[i] += loss.sum().item()
                    sum_biases[i] += bias.sum().item()
                sum_n += losses[0].size(0)
        
        sum_losses = torch.Tensor(sum_losses) / sum_n
        sum_biases = torch.Tensor(sum_biases) / sum_n
        return sum_losses, sum_biases

    def check_loss(it, time_t0, time_t1):
        sum_n = 0
        sum_loss_1 = 0
        sum_loss_2 = 0
        sum_loss_3 = 0
        sum_loss_4 = 0
        sum_bias_1 = 0
        sum_bias_2 = 0
        r1_ = []
        r2_ = []
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_loader):
                batch = batch.to(args.device)
                loss1, loss2, loss3, loss4, bias1, bias2, r1, r2 = model._check_loss(
                    atom_type=batch.atom_type,
                    r_feat=batch.r_feat,
                    p_feat=batch.p_feat,
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    t0=time_t0,
                    t1=time_t1,
                )

                sum_bias_1 += bias1.sum().item()
                sum_bias_2 += bias2.sum().item()
                sum_loss_1 += loss1.sum().item()
                sum_loss_2 += loss2.sum().item()
                sum_loss_3 += loss3.sum().item()
                sum_loss_4 += loss4.sum().item()
                sum_n += loss1.size(0)
                r1_.append(r1)
                r2_.append(r2)

        avg_bias_1 = sum_bias_1 / sum_n
        avg_bias_2 = sum_bias_2 / sum_n
        avg_loss_1 = sum_loss_1 / sum_n
        avg_loss_2 = sum_loss_2 / sum_n
        avg_loss_3 = sum_loss_3 / sum_n
        avg_loss_4 = sum_loss_4 / sum_n
        avg_r1 = torch.Tensor(r1_).mean().item()
        avg_r2 = torch.Tensor(r2_).mean().item()
        return (avg_loss_1, avg_loss_2, avg_loss_3, avg_loss_4, 
               avg_bias_1, avg_bias_2, avg_r1, avg_r2)

    def test_fn(it, time_t0, time_t1, extend_order, extend_radius):
        sum_n = 0
        sum_bias_1 = 0
        sum_bias_2 = 0
        sum_r1 = []
        sum_r2 = []
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_loader):
                batch = batch.to(args.device)
                bias1, bias2, r1, r2 = model._test_fn(
                    atom_type=batch.atom_type,
                    r_feat=batch.r_feat,
                    p_feat=batch.p_feat,
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_nodes_per_graph=None,
                    num_graphs=batch.num_graphs,
                    return_unreduced_loss=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    t0=time_t0,
                    t1=time_t1,
                )

                sum_bias_1 += bias1.sum().item()
                sum_bias_2 += bias2.sum().item()
                sum_r1.append(r1.item())
                sum_r2.append(r2.item())
                sum_n += bias1.size(0)
                
        avg_bias_1 = sum_bias_1 / sum_n
        avg_bias_2 = sum_bias_2 / sum_n
        avg_r1 = sum(sum_r1) / len(sum_r1)
        avg_r2 = sum(sum_r2) / len(sum_r2)
        return avg_bias_1, avg_bias_2, avg_r1, avg_r2
    
    def validate(it, time_t0, time_t1):
        sum_loss, sum_n = 0, 0
        sum_loss_global, sum_n_global = 0, 0
        sum_loss_local, sum_n_local = 0, 0
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_loader):
                batch = batch.to(args.device)
                #loss, loss_global, loss_local = model.get_loss(
                #    atom_type=batch.atom_type,
                #    r_feat=batch.r_feat,
                #    p_feat=batch.p_feat,
                #    pos=batch.pos,
                #    bond_index=batch.edge_index,
                #    bond_type=batch.edge_type,
                #    batch=batch.batch,
                #    num_nodes_per_graph=None,
                #    num_graphs=batch.num_graphs,
                #    return_unreduced_loss=True,
                #)
                loss, loss_global, loss_local = model.check_loss(
                    atom_type=batch.atom_type,
                    r_feat=batch.r_feat,
                    p_feat=batch.p_feat,
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    time_step=(time_t0, time_t1),
                    return_unreduced_loss=True,
                )
                sum_loss += loss.sum().item()
                sum_n += loss.size(0)
        avg_loss = sum_loss / sum_n
        return avg_loss

    try:
        it = 0
        for it in range(1):
            time_steps = [(1000*i, 1000*(i+1)) for i in range(5)]
            for t in time_steps:
                t0, t1 = t
                losses, biases = check_contribution(it, t0, t1)
                print(f"timesteps {t}")
                print("loss\t"+"\t".join([f"{x:0.4e}" for x in losses]))
                print("bias\t"+"\t".join([f"{x:0.4e}" for x in biases]))
            

    except KeyboardInterrupt:
        pass
    exit()

    #try:
    #    it = 0
    #    labels = ["bias1", "bias2", "edges/3N-6", "edges-cutoff/3N-6"]
    #    print("\t\t\t\t".join(labels))
    #    for ex_r in [1,2,3,4]:
    #        print(f"extended radius : {ex_r}")
    #        time_steps = [(1000*i, 1000*(i+1)) for i in range(5)]
    #        loss1_list = []
    #        loss2_list = []
    #        r1_list = []
    #        r2_list = []
    #        
    #        for t in time_steps:
    #            t0, t1 = t
    #            out = test_fn(it, t0, t1, ex_r, 10.0)
    #            msg = f"timestep {t}\t :" + "\t".join([f"{x:0.4f}" for x in out])
    #            print(msg)

    #except KeyboardInterrupt:
    #    logger.info("Terminating...")

    #try:
    #    it = 0
    #    labels = ["time_steps", "loss(b)", "loss(ub)", "loss-rad(b)", "loss-rad(ub)", "bias1", "bias2", "edges", "edges-rad"]
    #    print("\t\t".join(labels))
    #    for it in range(1):
    #        time_steps = [(1000*i, 1000*(i+1)) for i in range(5)]
    #        for t in time_steps:
    #            t0, t1 = t
    #            out = check_loss(it, t0, t1)
    #            msg = "\t\t".join([f"{t}"] + [f"{x:0.4e}" for x in out])
    #            print(msg)

    #        

    #except KeyboardInterrupt:
    #    #logger.info("Terminating...")
    #    pass

    exit()
    try:
        for it in range(start_iter, config.train.max_iters + 1):
            print(it)
            time_steps = [(1000*i, 1000*(i+1)) for i in range(5)]
            loss_list = []
            for t in time_steps:
                t0, t1 = t
                l = validate(it, t0, t1)
                loss_list.append(l)
            
            log = dict([(t[0], l) for t, l in zip(time_steps, loss_list)])
            if use_wandb:
                wandb.log(log)
            for k, v in log.items():
                print(f"{k} : {v}")

    except KeyboardInterrupt:
        logger.info("Terminating...")
        pass
