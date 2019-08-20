from qiskit_chemistry import FermionicOperator
from qiskit_chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.quantum_info import Pauli
from qiskit_aqua import Operator
from qiskit_aqua.algorithms import IQPE, ExactEigensolver
from qiskit import Aer,QuantumRegister
from qiskit import execute

import yaml as yml
from yaml import SafeLoader as Loader
import numpy as np
import Load_Hamiltonians as lh
from scipy import linalg as las

def construct_twoqubit_pauliham(hI,h0,h1,h2,h3,h4):
    return [[hI,Pauli(label='II')],[h0,Pauli(label='IZ')],[h1,Pauli(label='ZI')],[h2,Pauli(label='XX')],[h3,Pauli(label='YY')],[h4,Pauli(label='ZZ')]]

map_type = 'jordan_wigner'
backend = Aer.get_backend('qasm_simulator')
# backend = Aer.get_backend('unitary_simulator')

NW_data_file = str('/Users/mmetcalf/Dropbox/Quantum Embedding/Codes/Lithium_Downfolding/Qiskit Chem/HamiltonianDownfolding_w_IBM/H2.yaml')

print('First File', NW_data_file)

try:
    doc = open(NW_data_file, 'r')
    data = yml.load(doc, Loader)
finally:
    doc.close()

# Import all the data from a yaml file
print('Getting data')
n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
print('{} spatial orbitals'.format(n_spatial_orbitals))
nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
print('{} Coloumb repulsion'.format(nuclear_repulsion_energy))
n_orbitals = 2 * n_spatial_orbitals
n_particles = data['integral_sets'][0]['n_electrons']
print('{} particles'.format(n_particles))
dist = 2 * data['integral_sets'][0]['geometry']['atoms'][1]['coords'][2]
print('Bond distance is {}'.format(dist))
if map_type == 'parity':
    # For two-qubit reduction
    n_qubits = n_orbitals - 2
else:
    n_qubits = n_orbitals

# Importing the integrals
one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

# Getting spatial integrals and spin integrals to construct Hamiltonian
one_electron_spatial_integrals, two_electron_spatial_integrals = lh.get_spatial_integrals(one_electron_import,
                                                                                          two_electron_import,
                                                                                          n_spatial_orbitals)

h1, h2 = lh.convert_to_spin_index(one_electron_spatial_integrals,two_electron_spatial_integrals,n_spatial_orbitals, .01)

fop = FermionicOperator(h1, h2)
qop_paulis = fop.mapping(map_type)
# #print(qop_paulis._paulis)
# qop1 = Operator(paulis=qop_paulis.paulis)
# qop2 = Operator(paulis=qop_paulis.paulis)
#print(id(qop1),' and ', id(qop2))
init_state = HartreeFock(n_qubits, n_orbitals, n_particles, map_type, two_qubit_reduction=False)

#Hamiltonian from Google Paper
hI = -0.2265
h0 = 0.1843
h1 = -0.0549
h2 = 0.1165
h3 = 0.1165
h4 = 0.4386

t_0 = 9.830
#this function returns a list of pauli operators
google_pauliham = construct_twoqubit_pauliham(hI,h0,h1,h2,h3,h4)
google_op = Operator(paulis=google_pauliham)
google_init_state = HartreeFock(num_qubits=2, num_orbitals=2, num_particles=1, qubit_mapping=map_type, two_qubit_reduction=False)
# a = QuantumRegister(1, name='a')
# q = QuantumRegister(2, name='q')
# Circuit = google_op.construct_evolution_circuit(google_pauliham,t_0 , 1, q,
#                                     ancillary_registers=a, ctl_idx=0, unitary_power=None, use_basis_gates=False,
#                                     shallow_slicing=False)
# print(Circuit)
# result = execute(Circuit, backend).result()
# unitary = result.get_unitary(Circuit)
# print("Circuit unitary:\n", unitary)
# eval, evec = las.eigh(unitary)
# print(eval[0])

k = 7
#print('{} are the Paulis before'.format(qop2.paulis))
algorithm = IQPE(operator=google_op, state_in=google_init_state, num_time_slices=1, num_iterations=k, paulis_grouping='random', expansion_mode='trotter')
#print('New location of other op:', id(qop2))
#manually
# algorithm.identity_phase()
# qc_pea = algorithm.construct_circuit(k=1,omega=0)
# alg = execute(qc_pea, backend)
# result = alg.result()
# count = result.get_counts(qc_pea)
# print(qc_pea)
# print('{} are the state populations'.format(count))


result = algorithm.run(backend)
iqpe_energy = result['energy']
print('{} is the energy i got for {} iterations'.format(iqpe_energy,k))


# exact_eigensolver = ExactEigensolver(operator=google_op, k=2)
# ret = exact_eigensolver.run()
# print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real ))