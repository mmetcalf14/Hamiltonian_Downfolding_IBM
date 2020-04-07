from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry import QMolecule as qm
from qiskit.aqua.operators import Z2Symmetries, WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.providers.aer import noise
from qiskit.visualization import circuit_drawer

from qiskit.aqua.components.optimizers import COBYLA, L_BFGS_B
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit import Aer, IBMQ
from qiskit.compiler import transpile


# from pytket.qiskit import *
# from pytket import Transform

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import numpy as np
from scipy import linalg as la
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


def symmetries_5_8qbit_jwmap():
    symm = [Pauli(z=[True, False, False, False, False, True, True, True], x=[False, False, False, False, False, False, False, False]),
            Pauli(z=[False, True, False, False, False, True, False, False], x=[False, False, False, False, False, False, False, False]),
            Pauli(z=[False, False, True, False, False, False, True, False], x=[False, False, False, False, False, False, False, False]),
            Pauli(z=[False, False, False, True, False, False, False, True], x=[False, False, False, False, False, False, False, False]),
            Pauli(z=[False, False, False, False, True, True, True, True], x=[False, False, False, False, False, False, False, False])]

    sq_pauli = [Pauli(z=[False, False, False, False, False, False, False, False], x=[True, False, False, False, False, False, False, False]),
                Pauli(z=[False, False, False, False, False, False, False, False], x=[False, True, False, False, False, False, False, False]),
                Pauli(z=[False, False, False, False, False, False, False, False], x=[False, False, True, False, False, False, False, False]),
                Pauli(z=[False, False, False, False, False, False, False, False], x=[False, False, False, True, False, False, False, False]),
                Pauli(z=[False, False, False, False, False, False, False, False], x=[False, False, False, False, True, False, False, False])]

    sq_list = [0, 1, 2, 3, 4]

    return symm, sq_pauli, sq_list

# IBMQ.enable_account(token='f3e59f3967991477fc6a8413858524079a26a6e512874242a4ef41532cceca99b646d05af0e29fe234bd4aa5f4
#                            d491b6a5cd90734b124f5e8877d53884eafb74')
# provider = IBMQ.get_provider(hub='ibm-q-ornl')
# print(provider.backends())
# device = provider.get_backend('ibmq_boeblingen')
#################
backend1 = Aer.get_backend('statevector_simulator')
backend2 = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(shots=2**10, backend=backend1)


#Importing data generated from NW Chem to run experiments
root_dir = '/Users/mmetcalf/Dropbox/Quantum Embedding/Codes/Lithium_Downfolding/QiskitChem/Hamiltonian_Downfolding_IBM/IntegralData/H2_MEKENA/'
NW_data_file = str(root_dir+'h2_ccpvtz_ccsd_1_4008au_ducc_1_3.yaml')
# NW_data_file = str('H2.yaml')
OE_data_file = str(root_dir+ 'h2_ccpvtz_ccsd_1_4008au.FOCK')
# NW_data_file = 'H2.yaml'
doc_nw = open(NW_data_file,'r')
doc_oe = open(OE_data_file,'r')
data = yml.load(doc_nw,Loader)
content_oe = doc_oe.readlines()

###########################################################
#look at the hartree fock state and look at largest contri
#Initialize Variables
map_type = str('jordan_wigner')
n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
# nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
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
hf_energy = data['integral_sets'][0]['scf_energy']['value']
qm.hf_energy = hf_energy
# qm.hf_energy = molecule.hf_energy

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

###########################################################
one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

one_electron_spatial_integrals, two_electron_spatial_integrals = lh.get_spatial_integrals(one_electron_import,two_electron_import,n_spatial_orbitals)
qm.mo_eri_ints = two_electron_spatial_integrals

h1 = qm.onee_to_spin(one_electron_spatial_integrals)
h2 = qm.twoe_to_spin(two_electron_spatial_integrals)
print(np.shape(h2))
print('Getting Pauli Operator')
fop = FermionicOperator(h1, h2)
qop_paulis = fop.mapping(map_type)


if map_type == 'parity':
    two_qubit_reduction = True
    n_qubits = n_orbitals - 2
    qop_paulis = Z2Symmetries.two_qubit_reduction(qop_paulis, n_particles)
else:
    two_qubit_reduction = False
    n_qubits = n_orbitals

start_time = time.time()

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
init_state = HartreeFock(n_qubits, n_orbitals, n_particles, map_type,two_qubit_reduction=two_qubit_reduction)

var_op = UCCSD(num_qubits=n_qubits, depth=1, num_orbitals=n_orbitals, num_particles=n_particles, active_occupied=active_occ_list,\
               active_unoccupied=active_virt_list,initial_state=init_state, qubit_mapping=map_type, \
               two_qubit_reduction=two_qubit_reduction, mp2_reduction=False, singles_deletion=False)

print('There are {} params in this anzats'.format(var_op.num_parameters))
print(var_op._double_excitations)
dumpy_params = np.random.rand(var_op.num_parameters)
var_cirq = var_op.construct_circuit(dumpy_params)
print('Depth: ', var_cirq.depth())
# print(var_cirq)
# qasm_cirq = var_cirq.qasm()
#print(qasm_cirq)

# Fout = open('UCCSpD_QASM_HF_Orb-{}_Dist-{}.qasm'.format(n_spatial_orbitals,dist),"w")
# Fout.write(qasm_cirq)
# Fout.close()