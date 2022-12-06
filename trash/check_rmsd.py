import os
import numpy as np
from ase.io import read
from ase.build.rotate import minimize_rotation_and_translation

def get_dm(pos):
    n = len(pos)
    dm = np.linalg.norm(pos.reshape(1,n,3) - pos.reshape(n,1,3), axis=-1)
    return dm

def get_dm_mae(pos1, pos2):
    dm1 = get_dm(pos1)
    dm2 = get_dm(pos2)
    mae = (np.abs(dm1 - dm2)).mean()
    return mae

def get_mae(pos1, pos2):
    diff = pos1 - pos2
    dist = np.linalg.norm(diff, axis=-1)
    mae = np.mean(dist)
    return mae

def get_rmsd(pos1, pos2):
    diff = pos1 - pos2
    dist_squre = np.linalg.norm(diff, axis=-1)**2
    rmsd = np.sqrt(np.mean(dist_squre))
    return rmsd

def get_dm_rmsd(pos1, pos2):
    dm1 = get_dm(pos1)
    dm2 = get_dm(pos2)
    rmsd = np.sqrt(((dm1 - dm2)**2).mean())
    return rmsd

xyz_dir = "logs/ts_dv3_newedge_nolocal___dv3_newedge_nolocal/sample_ld_wg1.0/xyz_gen"
mae_all = []
dm_mae_all = []
rmsd_all = []
dm_rmsd_all = []

for i in range(1000):
    ref = os.path.join(xyz_dir, f"samples_{i}_ref.xyz")
    ref = read(ref)
    gen_list = []
    for j in [0 ,1]:
        file_name = f"samples_{i}_{j}.xyz"
        file_name = os.path.join(xyz_dir, file_name)
        gen_list.append(read(file_name))
    
    mae_list = []
    rmsd_list = []
    dm_mae_list = []
    dm_rmsd_list = []
    
    for gen in gen_list:
        minimize_rotation_and_translation(ref, gen)
        pos_gen = gen.get_positions()
        pos_ref = ref.get_positions()
        mae1 = get_mae(pos_ref, pos_gen)
        rmsd1 = get_rmsd(pos_ref, pos_gen)
        dm_mae = get_dm_mae(pos_ref, pos_gen)
        dm_rmsd = get_dm_rmsd(pos_ref, pos_gen)
        
        pos_gen[:,-1] = -pos_gen[:,-1]
        gen.set_positions(pos_gen)
        minimize_rotation_and_translation(ref, gen)
        pos_gen = gen.get_positions()
        pos_ref = ref.get_positions()
        mae2 = get_mae(pos_ref, pos_gen)
        rmsd2 = get_rmsd(pos_ref, pos_gen)
        mae = min(mae1, mae2)
        rmsd = min(rmsd1, rmsd2)
        #mae = mae1
        #rmsd = rmsd1
         
        mae_list.append(mae)
        rmsd_list.append(rmsd)
        dm_mae_list.append(dm_mae)
        dm_rmsd_list.append(dm_rmsd)
    
    mae_all.append(mae_list)
    dm_mae_all.append(dm_mae_list)
    rmsd_all.append(rmsd_list)
    dm_rmsd_all.append(dm_rmsd_list)

rmsd_all = np.array(rmsd_all) 
dm_rmsd_all = np.array(dm_rmsd_all)
mae_all = np.array(mae_all) 
dm_mae_all = np.array(dm_mae_all)

np.savez("score_nolocal", mae=mae_all, rmsd=rmsd_all, dm_mae=dm_mae_all, dm_rmsd=dm_rmsd_all)



#xyz_dir = "logs/ts_dv3_newedge_base___dv3_newedge_base/sample_ld_wg1.0/xyz_gen"
#mae_all = []
#dm_mae_all = []
#
#for i in range(1000):
#    ref = os.path.join(xyz_dir, f"samples_{i}_ref.xyz")
#    ref = read(ref)
#    gen_list = []
#    for j in [0 ,1]:
#        file_name = f"samples_{i}_{j}.xyz"
#        file_name = os.path.join(xyz_dir, file_name)
#        gen_list.append(read(file_name))
#    
#    mae_list = []
#    dm_mae_list = []
#    
#    for gen in gen_list:
#        minimize_rotation_and_translation(ref, gen)
#        pos_gen = gen.get_positions()
#        pos_ref = ref.get_positions()
#        mae1 = get_mae(pos_ref, pos_gen)
#        dm_mae = get_dm_mae(pos_ref, pos_gen)
#        
#        pos_gen[:,-1] = -pos_gen[:,-1]
#        gen.set_positions(pos_gen)
#        minimize_rotation_and_translation(ref, gen)
#        pos_gen = gen.get_positions()
#        pos_ref = ref.get_positions()
#        mae2 = get_mae(pos_ref, pos_gen)
#        
#        mae = min(mae1, mae2)
#        mae_list.append(mae)
#        dm_mae_list.append(dm_mae)
#    
#    mae_all.append(mae_list)
#    dm_mae_all.append(dm_mae_list)
#mae_all = np.array(mae_all) 
#dm_mae_all = np.array(dm_mae_all)
#
#np.savez("score_base", mae=mae_all, dm_mae=dm_mae_all)
