from Basis import Basis
import numpy as np

Norb =10
Nup = 4
Ndn = 4
my_basis = Basis(Norb,Nup,Ndn, basis_type='block')

#basis = my_basis.create_single_particle_basis(Norb,Nup)
# basis = my_basis.create_total_basis()
# singles, doubles = my_basis.compute_varform_excitations(basis)
# print(doubles)

gs = np.arange(Nup)
print(gs)

sum = 0

