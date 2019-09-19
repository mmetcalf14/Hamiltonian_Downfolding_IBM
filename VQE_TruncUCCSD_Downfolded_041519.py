from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry import QMolecule as qm
from qiskit.aqua.operators import Z2Symmetries, WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.visualization import circuit_drawer

from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua import Operator
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit import Aer


from pytket.qiskit import *
from pytket import Transform

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import numpy as np
import yaml as yml
from yaml import SafeLoader as Loader
import Load_Hamiltonians as lh
import os
import time


def symmetries_4_8qbit_jwmap():

    symm = [Pauli(z=[True, True, False, False, False, False, True, True],
           x=[False, False, False, False, False, False, False, False]),
     Pauli(z=[False, False, True, False, False, False, True, False],
           x=[False, False, False, False, False, False, False, False]),
     Pauli(z=[False, False, False, True, False, False, False, True],
           x=[False, False, False, False, False, False, False, False]),
     Pauli(z=[False, False, False, False, True, True, True, True],
           x=[False, False, False, False, False, False, False, False])]

    sq_pauli = [Pauli(z=[False, False, False, False, False, False, False, False],
                      x=[True, False, False, False, False, False, False, False]),
                Pauli(z=[False, False, False, False, False, False, False, False],
                      x=[False, False, True, False, False, False, False, False]),
                Pauli(z=[False, False, False, False, False, False, False, False],
                      x=[False, False, False, True, False, False, False, False]),
                Pauli(z=[False, False, False, False, False, False, False, False],
                      x=[False, False, False, False, True, False, False, False])]

    sq_list = [0, 2, 3, 4]
    return symm, sq_pauli, sq_list

#Importing data generated from NW Chem to run experiments
#Li2_cc-pVTZ_4_ORBITALS_Li2-4_ORBITALS-ccpVTZ-2_1384
root_dir = '/Users/mmetcalf/Dropbox/Quantum Embedding/Codes/Lithium_Downfolding/Qiskit Chem/Hamiltonian_Downfolding_IBM/IntegralData/H2_MEKENA/'
NW_data_file = str(root_dir+'h2_ccpvtz_ccsd_1_4008au_ducc_1_3.yaml')
# NW_data_file = str('H2.yaml')
OE_data_file = str(root_dir+ 'h2_ccpvtz_ccsd_1_4008au.FOCK')
# NW_data_file = 'H2.yaml'
doc_nw = open(NW_data_file,'r')
doc_oe = open(OE_data_file,'r')
data = yml.load(doc_nw,Loader)
content_oe = doc_oe.readlines()
#################
backend1 = Aer.get_backend('statevector_simulator')
backend2 = Aer.get_backend('qasm_simulator')
#################

###########################################################
#Initialize Variables
map_type = str('jordan_wigner')
truncation_threshold = 0.1
n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
print('repulsion: ',nuclear_repulsion_energy)
n_orbitals = 2*n_spatial_orbitals
n_particles = data['integral_sets'][0]['n_electrons']

#Populating the QMolecule class with the data to make calculations easier
qm.num_orbitals = n_spatial_orbitals
qm.num_alpha = n_particles//2
qm.num_beta = n_particles//2
#This is just a place holder for freezing the core that I need for the MP2 calculations
qm.core_orbitals = 0
qm.nuclear_repulsion_energy = nuclear_repulsion_energy
qm.hf_energy = data['integral_sets'][0]['scf_energy']['value']

###################Get Orbital Energies from FOCK file########################
orbital_energies = []
found = False
count = 0
for line in content_oe:
    if 'Eigenvalues:' in line.split()[0]:
        found =True
    if found and len(line.split()) > 1:
        orbital_energies.append(float(line.split()[1]))
        count += 1
        if count >= n_spatial_orbitals:
            break
qm.orbital_energies = orbital_energies

#7 orbitals
#qm.orbital_energies = [-2.4533,-2.4530,-0.1817,0.0065, 0.0273, 0.0273, 0.0385]
##############################################################################

dist = 2*data['integral_sets'][0]['geometry']['atoms'][1]['coords'][2]

active_occ = n_particles//2
active_virt = (n_orbitals-n_particles)//2
active_occ_list = np.arange(active_occ)
active_occ_list = active_occ_list.tolist()
#print('Orbitals {} are occupied'.format(active_occ_list))
active_virt_list = np.arange(active_virt)
active_virt_list = active_virt_list.tolist()
#print('Orbitals {} are unoccupied'.format(active_virt_list))
###########################################################
#Output Files to Plot stuff
#Fout = open('ErrorFromTruncation_Li2_Orb-{}_Dist-{}'.format(n_spatial_orbitals,dist),"w")
###########################################################

one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

one_electron_spatial_integrals, two_electron_spatial_integrals = lh.get_spatial_integrals(one_electron_import,two_electron_import,n_spatial_orbitals)
one_electron_spatial_integrals, two_electron_spatial_integrals = lh.trunctate_spatial_integrals(one_electron_spatial_integrals,two_electron_spatial_integrals,.001)
print(one_electron_spatial_integrals)
#print('Spatial Integrals from my program\n',one_electron_spatial_integrals)

#print('The difference\n',molecule.mo_eri_ints-two_electron_spatial_integrals)
#For the MP2 calculation
qm.mo_eri_ints = two_electron_spatial_integrals
# qm.mo_onee_ints = one_electron_spatial_integrals

h1, h2 = lh.convert_to_spin_index(one_electron_spatial_integrals,two_electron_spatial_integrals,n_spatial_orbitals, .001)
#print(h1)
# h1 = qm.onee_to_spin(one_electron_spatial_integrals)
# h2 = qm.twoe_to_spin(np.einsum('ijkl->ljik', two_electron_spatial_integrals))

#fop = FermionicOperator(one_temp,two_temp)
fop = FermionicOperator(h1, h2)
qop_paulis = fop.mapping(map_type)

if map_type == 'parity':
    two_qubit_reduction = True
    n_qubits = n_orbitals - 2
    qop_paulis = Z2Symmetries.two_qubit_reduction(qop_paulis, n_particles)
else:
    two_qubit_reduction = False
    n_qubits = n_orbitals

# z2_symmetries = Z2Symmetries.find_Z2_symmetries(qop_paulis)
symm, sq_pauli, sq_list = symmetries_4_8qbit_jwmap()
z2_symmetries = Z2Symmetries(symmetries=symm, sq_paulis=sq_pauli, sq_list=sq_list)

print('Z2 symmetries found:')
for symm in z2_symmetries.symmetries:
    print(symm.to_label())
print('single qubit operators found:')
for sq in z2_symmetries.sq_paulis:
    print(sq.to_label())
print('cliffords found:')
for clifford in z2_symmetries.cliffords:
    print(clifford.print_details())
print('single-qubit list: {}'.format(z2_symmetries.sq_list))

tapered_ops = z2_symmetries.taper(qop_paulis)
for tapered_op in tapered_ops:
    print("Number of qubits of tapered qubit operator: {}".format(tapered_op.num_qubits))
smallest_eig_value = 99999999999999
smallest_idx = -1
for idx in range(len(tapered_ops)):
    ee = ExactEigensolver(tapered_ops[idx], k=1)
    curr_value = ee.run()['energy']
    if curr_value < smallest_eig_value:
        smallest_eig_value = curr_value
        smallest_idx = idx
    print("Lowest eigenvalue of the {}-th tapered operator (computed part) is {:.12f}".format(idx, curr_value))

the_tapered_op = tapered_ops[smallest_idx]
the_coeff = tapered_ops[smallest_idx].z2_symmetries.tapering_values
print(
    "The {}-th tapered operator matches original ground state energy, with corresponding symmetry sector of {}".format(
        smallest_idx, the_coeff))
###########################################
#Exact Result to compare to: This can also be obtained via the yaml file once we verify correctness.
#Don't use trunctated Hamiltonian
print('Getting energy')
exact_eigensolver = ExactEigensolver(qop_paulis, k=1)
ret = exact_eigensolver.run()
print('The electronic energy is: {:.12f}'.format(ret['eigvals'][0].real))
# print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real + nuclear_repulsion_energy))
###########################################

start_time = time.time()

# init_state = HartreeFock(n_qubits, n_orbitals, n_particles, map_type,two_qubit_reduction=two_qubit_reduction)
# #Keep in mind Jarrod didn't use any singles for his ansatz. Maybe just stick to some local doubles, diagonal cancel
# var_op = UCCSD(num_qubits=n_qubits, depth=1, num_orbitals=n_orbitals, num_particles=n_particles, active_occupied=active_occ_list,\
#                active_unoccupied=active_virt_list,initial_state=init_state, qubit_mapping=map_type, two_qubit_reduction=two_qubit_reduction, mp2_reduction=True, singles_deletion=False)

# Unitaries with symmetries
init_state = HartreeFock(num_qubits=the_tapered_op.num_qubits, num_orbitals=n_orbitals, num_particles=n_particles,
                         qubit_mapping=map_type,two_qubit_reduction=two_qubit_reduction, sq_list=the_tapered_op.z2_symmetries.sq_list)

var_op = UCCSD(num_qubits=the_tapered_op.num_qubits, depth=1, num_orbitals=n_orbitals, num_particles=n_particles, active_occupied=active_occ_list,\
               active_unoccupied=active_virt_list,initial_state=init_state, qubit_mapping=map_type, two_qubit_reduction=two_qubit_reduction,
               mp2_reduction=True, singles_deletion=True, z2_symmetries=the_tapered_op.z2_symmetries)
# print('There are {} params in this anzats'.format(var_op.num_parameters))
# print(var_op._double_excitations)
# dumpy_params = np.random.rand(var_op.num_parameters)
# var_cirq = var_op.construct_circuit(dumpy_params)
# print(var_cirq)

#Optimize
# tk_cirq = qiskit_to_tk(var_cirq)
# print('{} depth of unpotimized UCCSD circuit'.format(tk_cirq.depth()))
# Transform.OptimisePhaseGadgets().apply(tk_cirq)
# print('{} depth of UCCSD circuit with redundancies removed'.format(tk_cirq.depth()))
# var_cirq = tk_to_qiskit(tk_cirq)
# circuit_drawer(var_cirq, filename='uccsd_circuit_h2_jordanwigner_symm_091819.png',output='mpl')


# setup a classical optimizer for VQE
max_eval = 200
# Would using a different optimizer yield better results at long distances?
optimizer = COBYLA(maxiter=max_eval,disp=True, tol=1e-2)
print('params: ',var_op.num_parameters)
# dumpy_params = np.random.rand(var_op.num_parameters)
#Call the VQE algorithm class with your qubit operator for the Hamiltonian and the variational op
print('Doing VQE')
# algorithm = VQE(qop_paulis,var_op,optimizer,'paulis', initial_point=None)
algorithm = VQE(the_tapered_op,var_op,optimizer,'paulis', initial_point=None)
# VQE_Circ = algorithm.construct_circuit(dumpy_params, backend=None)
# print('The VQE circuit:\n',circuit_drawer(VQE_Circ, output='text'))
result = algorithm.run(backend1)
print('The VQE energy is: ',result['energy']+nuclear_repulsion_energy)
opt_params = result['opt_params']
print(' {} are the optimal parameters.'.format(opt_params))
end_time = time.time()
print('It took {} seconds to run this calculation'.format(end_time-start_time))

'''

#Fout.close()

# Junk Code I don't want to delete
# Determining a method to reduce the excitation list with MP2
# All I am missing are the orbital energies and then I am good to go
# It might be worth integrating this into the UCCSD class, so you can get the right circuit
# without having to redo it.
# mp2 = MP2Info(qm)
# mp2_coeff, new_energies = mp2.mp2_get_term_info(var_op._double_excitations)
#
# print('{} are the new MP2 coeffs with energies {}'.format(mp2_coeff,new_energies))
# new_doubles = []
# for i, val in enumerate(doubles):
#     if mp2_coeff[i] != 0:
#         new_doubles.append(val)

#print('These are the new doubles {}'.format(new_doubles))

'''



