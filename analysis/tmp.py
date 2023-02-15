from ase.visualize import view
from ase.io import read
import py3Dmol as p3d

viewer = p3d.view()
viewer.addModel("tmp.xyz","xyz")
viewer.show()
#atoms = read("tmp.xyz")
#view(atoms)
