import numpy as np
from basis import Basis
from itertools import combinations_with_replacement, combinations, product

Norb = 28
Nup = 1
Ndn = 1
my_basis = Basis(Norb,Nup,Ndn, basis_type='block')

#basis = my_basis.create_single_particle_basis(Norb,Nup)
basis = my_basis.create_total_basis()

print(basis)

# orbs = np.arange(0,Norb)
# max_iter = 0
# active_occ = Nup
# active_unocc = Norb-Nup
# print('range: ',range(Nup,Norb))
# if active_occ >= active_unocc:
#     max_iter = active_unocc
# else:
#     max_iter = active_occ
#
# excitations={}
# for it in range(1,(max_iter+1)):
#     print(it)
#     list = []
#     for vec1 in product(range(active_occ),repeat=it):
#         temp_vec1 = np.ndarray.tolist(np.array(vec1))
#         if (len(set(vec1)) > it-1) and temp_vec1 == sorted(temp_vec1):
#             occ_ind = np.array(vec1)
#             for vec2 in product(range(active_occ,Norb),repeat=it):
#                 temp_vec2 = np.ndarray.tolist(np.array(vec2))
#                 if len(set(vec2)) > it-1 and temp_vec2 == sorted(temp_vec2):
#                     unocc_ind = np.array(vec2)
#                     list.append(np.append(vec1,vec2))
#     excitations[it] = list
# print(excitations)
# list = []
# for i in product(range(Nup+1),repeat=iter):
#     print(set(i), len(i)) # for uniqueness
#     if len(set(i)) > iter-1:
#         list.append(np.array(i))
# print(list)
# comb_orb = combinations_with_replacement(orbs,2*iter)
# comb_orb = list(comb_orb)
#
# max_occ = 2
#
# new_list = []
# print((0.5*max_occ*(max_occ+1)))
# for i in comb_orb:
#
#     occ = i[0:iter]
#     print(occ)
#     print(list(combinations(occ,len(occ))))
#     #print(occ)
#     unocc = i[1]
#     x = int(np.sum(occ))
#     #print(x)
#     if x < (0.5*max_occ*(max_occ+1)-1):
#         new_list.append(i)
# print(new_list)