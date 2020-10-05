from Basis import Basis
import numpy as np

Norb =4
Nup = 1
Ndn = 1
my_basis = Basis(Norb,Nup,Ndn, basis_type='block')

# basis = my_basis.create_single_particle_basis(Norb,Nup)
basis = my_basis.create_total_basis()
index = []
for key, val in enumerate(basis):
    index.append(val)
print(index)
singles, doubles = my_basis.compute_varform_excitations(basis)
excitations = [singles,doubles]
print(excitations[0])
index = []
for key, val in enumerate(excitations[0]):
    index.append(val[1])
print(index)
# gs = np.arange(Nup)
# print(gs)
#
# sum = 0

