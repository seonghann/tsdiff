# import glob
import os
import pickle
import numpy as np
import argparse

import ase
import ase.io
from ase.build.rotate import minimize_rotation_and_translation
import rdkit.Chem as Chem

# import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import itertools


def position_align(ref_atoms, atoms_list):
    ret_atoms = []
    for prb_atom in atoms_list:
        ret_atoms.append(rotate_transform_mirror(ref_atoms, prb_atom))
    return ret_atoms


def rotate_transform_mirror(ref_atoms, prb_atoms):
    mirror_transform = np.eye(3)
    mirror_transform[2, 2] = -1

    ref_pos = ref_atoms.positions.copy()
    minimize_rotation_and_translation(ref_atoms, prb_atoms)
    p1 = prb_atoms.positions.copy()

    prb_atoms.positions = prb_atoms.positions @ mirror_transform
    minimize_rotation_and_translation(ref_atoms, prb_atoms)
    p2 = prb_atoms.positions.copy()

    rmsd1 = np.sqrt((np.linalg.norm(p1 - ref_pos, axis=1) ** 2).mean())
    rmsd2 = np.sqrt((np.linalg.norm(p2 - ref_pos, axis=1) ** 2).mean())
    if rmsd1 < rmsd2:
        prb_atoms.positions = p1
    else:
        prb_atoms.positions = p2
    return prb_atoms


def index_align(ref_atoms, atoms_list, smarts):
    ref_pos = ref_atoms.positions.copy()
    ret_atoms = []
    for i, prb_atoms in enumerate(atoms_list):
        matches = get_substruct_matches(smarts)
        prb_pos = prb_atoms.positions.copy()
        prb_an = prb_atoms.get_atomic_numbers()

        match, init_target, final_target = get_min_dmae_match(matches, ref_pos, prb_pos)
        ret_pos = prb_pos[match]
        ret_an = prb_an[match]
        ret_atoms.append(ase.Atoms(ret_an, positions=ret_pos))
    return ret_atoms


def get_min_dmae_match(matches, ref_pos, prb_pos):
    dmaes = []
    for match in matches:
        match_pos = prb_pos[list(match)]
        dmae = calc_DMAE(cdist(ref_pos, ref_pos), cdist(match_pos, match_pos))
        dmaes.append(dmae)
    return list(matches[dmaes.index(min(dmaes))]), dmaes[0], min(dmaes)


def get_substruct_matches(smarts):
    smarts_r, smarts_p = smarts.split(">>")
    mol_r = Chem.MolFromSmarts(smarts_r)
    mol_p = Chem.MolFromSmarts(smarts_p)

    matches_r = list(mol_r.GetSubstructMatches(mol_r, uniquify=False))
    map_r = np.array([atom.GetAtomMapNum() for atom in mol_r.GetAtoms()]) - 1
    map_r_inv = np.argsort(map_r)
    for i in range(len(matches_r)):
        matches_r[i] = tuple(map_r[np.array(matches_r[i])[map_r_inv]])

    matches_p = list(mol_p.GetSubstructMatches(mol_p, uniquify=False))
    map_p = np.array([atom.GetAtomMapNum() for atom in mol_p.GetAtoms()]) - 1
    map_p_inv = np.argsort(map_p)
    for i in range(len(matches_p)):
        matches_p[i] = tuple(map_p[np.array(matches_p[i])[map_p_inv]])

    matches = set(matches_r) & set(matches_p)
    matches = list(matches)
    matches.sort()
    return matches


def calc_DMAE(dm_ref, dm_guess, mape=False):
    if mape:
        retval = abs(dm_ref - dm_guess) / dm_ref
    else:
        retval = abs(dm_ref - dm_guess)
    # retval = torch.triu(retval, diagonal=1).sum() / len(dm_ref) / (len(dm_ref) - 1) * 2
    retval = np.triu(retval, k=1).sum() / len(dm_ref) / (len(dm_ref) - 1) * 2
    return retval


def convert_from_xyz(xyz_file, smarts, index_list):
    atoms = list(ase.io.iread(xyz_file, index=":"))
    prb_atoms = [atoms[i] for i in index_list]
    ref_atoms = prb_atoms[0]
    # first, align the atom index according to the first frame
    prb_atoms = convert_from_atoms(ref_atoms, prb_atoms, smarts)
    return prb_atoms


def convert_from_atoms(ref_atoms, prb_atoms, smarts):
    prb_atoms = index_align(ref_atoms, prb_atoms, smarts)
    prb_atoms = position_align(ref_atoms, prb_atoms)
    return prb_atoms


def get_minimum_matches(ref, prb, matches=[], metric=lambda du, dv: ((du - dv) ** 2).sum(), return_type="value"):
    # u, v is atom positions, (n, 3)
    # matches is a list of permutations of atom indices
    metric_bins = []
    d_ref = pdist(ref)
    for match in matches:
        d_prb = pdist(prb[match])
        metric_bins.append(metric(d_ref, d_prb))
    if return_type == "value":
        return min(metric_bins)
    else:
        min_match = matches[metric_bins.index(min(metric_bins))]
        return min_match


def adjust_color_brightness(color, brightness_factor):
    """
    :param color: input color string (e.g: 'red', 'blue', '#FF5733', ...)
    :param brightness_factor: brighter when > 1, darker when < 1 (float)
    :return: return color string
    """
    rgba_color = mcolors.to_rgba(color)
    adjusted_color = [min(max(channel * brightness_factor, 0.0), 1.0) for channel in rgba_color[:3]]
    adjusted_rgba = tuple(adjusted_color + [rgba_color[3]])  # 원래의 alpha 값 유지
    return mcolors.to_hex(adjusted_rgba)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh", type=float, default=0.10)
    parser.add_argument("--sample_index", type=int, default=0)

    parser.add_argument("--save_dir", type=str, default="clustering")
    parser.add_argument("--sample_path", type=str, default="generated/samples_all.pkl")
    parser.add_argument("--num_levels", type=int, default=3)

    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    thresh = args.thresh

    # load generated data
    gen_data = pickle.load(open(args.sample_path, "rb"))
    smarts = gen_data[args.sample_index].smiles
    gen_data = [d for d in gen_data if d.smiles == smarts]
    gen_atoms = []
    for d in gen_data:
        if d.pos_gen.ndim == 3:
            pos = d.pos_gen[-1].numpy()
        else:
            pos = d.pos_gen.numpy()
        an = d.atom_type.numpy()
        gen_atoms.append(ase.Atoms(positions=pos, numbers=an))
    num_gen = len(gen_atoms)

    # hierarcy clustering
    def f(u, v, smarts=smarts):
        matches = np.array(get_substruct_matches(smarts))
        u = u.copy().reshape(-1, 3)
        v = v.copy().reshape(-1, 3)

        def metric_(du, dv):
            return np.sqrt(((du - dv) ** 2).mean())

        ret = get_minimum_matches(u, v, matches=matches, metric=metric_, return_type="value")
        return ret

    pos_flat = np.array([atom.positions for atom in gen_atoms]).reshape(len(gen_atoms), -1)
    print("start clustering")
    linkage_matrix = linkage(pos_flat, "single", optimal_ordering=True, metric=f)
    label_list = range(1, num_gen + 1)
    clusters = fcluster(linkage_matrix, t=thresh, criterion='distance')
    num_clusters = max(clusters)

    # Draw figure
    print("start drawing")
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    label_list = ["(1)"] * num_gen
    # Dendrogram
    ax = axes[0]
    num_levels = args.num_levels
    dendrogram(
        linkage_matrix, num_levels,
        truncate_mode="level",
        color_threshold=thresh,
        orientation='top',
        labels=label_list,
        distance_sort='descending',
        show_leaf_counts=True,
        above_threshold_color="k",
        ax=ax
    )
    ax.hlines(thresh, 0, 2000, color="k", linestyle="--", alpha=0.7)

    if os.path.isdir(args.save_dir):
        if args.force:
            os.system(f"rm -r {args.save_dir}")
        else:
            raise ValueError(f"{args.save_dir} already exists. Use --force to overwrite.")
            exit()
    os.system(f"mkdir -p {args.save_dir}")

    plt.savefig(os.path.join(args.save_dir, "hierarchy_clustering.png"))
    stat = {
        "num_clusters": len(set(clusters)),
        "cluster": clusters,
        "dist_mat": cdist(pos_flat, pos_flat, metric=f),
    }

    with open(os.path.join(args.save_dir, "stat_clustering.pkl"), "wb") as f:
        pickle.dump(stat, f)

    print("start converting xyz for saving")
    # write xyz files based on clustering results.
    # delete existing files first
    # permute and align representative atoms in each cluster to the reference
    for i in range(1, num_clusters + 1):
        rep_atoms_idx = np.where(clusters == i)[0][0]
        rep_atoms = gen_atoms[rep_atoms_idx]
        rep_atoms = convert_from_atoms(gen_atoms[0], [rep_atoms], smarts)
        gen_atoms[rep_atoms_idx] = rep_atoms[0]

    for i in range(1, num_clusters + 1):
        cluster_i_indices = np.where(clusters == i)[0]
        gen_atoms_cluster_i = [gen_atoms[j] for j in cluster_i_indices]
        rep_atoms = gen_atoms_cluster_i[0]
        save_atoms = convert_from_atoms(rep_atoms, gen_atoms_cluster_i, smarts)

        for atoms in save_atoms:
            ase.io.write(f"{args.save_dir}/cluster_{i}.xyz", atoms, append=True)
