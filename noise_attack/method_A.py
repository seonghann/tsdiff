import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import pandas as pd
import numpy as np
import torch
import random


def read_xyz_file(r_file, p_file):
    with open(r_file,"r") as f:
        r_lines = f.readlines()
    with open(p_file,"r") as f:
        p_lines = f.readlines()
    r_xyz_blocks = []
    p_xyz_blocks = []
    N = 0
    for l in r_lines:
        try:
            n = int(l.strip())+2
            r_xyz_blocks.append("".join(r_lines[N:N+n]))
            N += n
        except:
            pass
    
    N = 0
    for l in p_lines:
        try:
            n = int(l.strip())+2
            p_xyz_blocks.append("".join(p_lines[N:N+n]))
            N += n
        except:
            pass

    return r_xyz_blocks, p_xyz_blocks

def xyz_block_to_pos(block):
    lines = block.strip().split("\n")
    xyz = []
    for l in lines[2:]:
        a_xyz = [float(v) for v in l.strip().split("\t")[1:]]
        xyz.append(a_xyz)
    return torch.DoubleTensor(xyz)

def reform_xyz_block(xyz_block, new_pos :torch.Tensor):
    lines = xyz_block.strip().split("\n")
    new_lines = lines[:2]

    assert len(new_pos) == len(lines[2:])
    for l, a_pos in zip(lines[2:], new_pos):
        a_pos = "\t".join([str(i.item()) for i in a_pos])
        a_line = "\t".join([l.split("\t")[0], a_pos])
        new_lines.append(a_line)
    new_xyz_block = "\n".join(new_lines)
    return new_xyz_block

def get_axis_from_com(smarts, pos):
    mol = Chem.MolFromSmiles(smarts)
    mapnum = []
    for a in mol.GetAtoms():
        mapnum.append(a.GetAtomMapNum())
    map_mask = np.array([i+1 in mapnum for i in range(len(pos))])
   
def get_axis_from_bond(r_smarts, p_smarts, r_pos, p_pos, alpha=0.2):
    if not "." in r_smarts and not "." in p_smarts:
        return 
    
    if len(r_smarts.split(".")) > 2 or len(p_smarts.split(".")) > 2:
        return
    
    if "." in r_smarts:
        forward = True
    else:
        forward = False

    mol_r = Chem.MolFromSmiles(r_smarts)
    mol_p = Chem.MolFromSmiles(p_smarts)
    r_bd = [set([
        b.GetBeginAtom().GetAtomMapNum(), 
        b.GetEndAtom().GetAtomMapNum()
        ]) for b in mol_r.GetBonds()]
    p_bd = [set([
        b.GetBeginAtom().GetAtomMapNum(), 
        b.GetEndAtom().GetAtomMapNum()
        ]) for b in mol_p.GetBonds()]
    

    bond_forming = []
    if forward:
        bi_bd = r_bd
        uni_bd = p_bd
    else:
        bi_bd = p_bd
        uni_bd = r_bd

    for i in uni_bd:
        if i in p_bd:
            continue
        bond_forming.append(i)
    
    bi_smarts = r_smarts if forward else p_smarts
    bi_pos = r_pos if forward else p_pos
    random.shuffle(bond_forming)

    i, j = [x-1 for x in list(bond_forming[0])]
    vec = bi_pos[i] - bi_pos[j]
    
    smi1, smi2 = bi_smarts.split(".")
    mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    index1 = [a.GetAtomMapNum() for a in mol1.GetAtoms()]
    index2 = [a.GetAtomMapNum() for a in mol2.GetAtoms()]
    N = len(index1) + len(index2)

    mapnum = index1 if j+1 in index1 else index2
    map_mask = np.array([i+1 in mapnum for i in range(N)])
    
    vec = vec/vec.norm()
    bi_pos[map_mask] += vec * alpha
    
    return bond_forming


if __name__ == "__main__":
    r_path = "/home/ksh/MolDiff/GeoDiff/data/TS/b97d3/random_split/raw_data/b97d3_r_test.xyz"
    p_path = "/home/ksh/MolDiff/GeoDiff/data/TS/b97d3/random_split/raw_data/b97d3_p_test.xyz"
    r_blocks, p_blocks = read_xyz_file(r_path, p_path)

    r_xyz = r_blocks[0]
    p_xyz = p_blocks[0]
    r_lines = r_xyz.strip().split("\n")
    p_lines = p_xyz.strip().split("\n")
    r_smarts = r_lines[1].strip()
    p_smarts = p_lines[1].strip()
    print(r_smarts)
    print(p_smarts)

    # check if multi-molecular reaction
    #
    #

    r_pos = xyz_block_to_pos(r_xyz)
    p_pos = xyz_block_to_pos(p_xyz)

