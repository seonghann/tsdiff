
def parse_xyz(filename):
    """Parse xyz file and return a list of atoms and their coordinates."""
    atoms = []
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().isdigit():
                continue
            atom, x, y, z = line.split()
            atoms.append(atom)
            coords.append((float(x), float(y), float(z)))
    return atoms, coords


def parse_xyz_block(xyz_block):
    """Parse a xyz block and return a list of atoms and their coordinates."""
    atoms = []
    coords = []
    for line in xyz_block:
        if line.strip().isdigit():
            continue
        atom, x, y, z = line.split()
        atoms.append(atom)
        coords.append((float(x), float(y), float(z)))
    return atoms, coords


def parse_xyz_corpus(filename):
    """
    Parse a long xyz file which is a sequence of xyz-block connected without seperator.
    Return a list of xyz-blocks.
    Each xyz-block contains a molecule.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    # First line is the number of atoms
    # Second line is the comment line
    # From third line, each line contains an atom and its coordinates
    # Gather lines to create a xyz-block
    xyzs = []
    i = 0
    while i < len(lines):
        n_atoms = int(lines[i])
        xyz_block = lines[i: i + n_atoms + 2]
        xyzs.append("".join(xyz_block).strip())
        i += n_atoms + 2
    return xyzs


if __name__ == '__main__':
    xyzs = parse_xyz_corpus('wb97xd3_ts.xyz')
    print(len(xyzs))
    print("------------------")
    print(xyzs[0])
    print("------------------")
    print(xyzs[1])
    print("------------------")
    print(xyzs[-1])
