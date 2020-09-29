from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry import QMolecule as qm
from qiskit.aqua.components.optimizers import COBYLA, L_BFGS_B, DIRECT_L, CRS, DIRECT_L_RAND, ESCH
from qiskit.aqua import QuantumInstance
from qiskit.chemistry.algorithms import VQEAdapt
from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit.aqua.operators.expectations import PauliExpectation, AerPauliExpectation
from qiskit import Aer, BasicAer

import numpy as np
import yaml as yml
from yaml import SafeLoader as Loader
import Load_Hamiltonians as lh
import time


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




# IBMQ.enable_account(token='f3e59f3967991477fc6a8413858524079a26a6e512874242a4ef41532cceca99b646d05af0e29fe234bd4aa5f4
#                            d491b6a5cd90734b124f5e8877d53884eafb74')
# provider = IBMQ.get_provider(hub='ibm-q-ornl')
# print(provider.backends())
# device = provider.get_backend('ibmq_boeblingen')
#################
#backend1 = Aer.get_backend('statevector_sipmulator')
backend2 = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(shots=2**10, backend=backend2)
#################

#Importing data generated from NW Chem to run experiments
root_dir = '/Users/mmetcalf/Dropbox/Quantum Embedding/Codes/Lithium_Downfolding/Qiskit Chem/Hamiltonian_Downfolding_IBM/IntegralData/BeH2 RHF/'
NW_data_file = str(root_dir+'YAML/BeH2-E1.yaml')
OE_data_file = str(root_dir+ 'FOCK/BeH2-RHF-DUCC-E1.FOCK')
# NW_data_file = str('H2O_4Virt_2.5Eq.yaml')
# OE_data_file = str('H2O_4Virt_2.5Eq.FOCK')
doc_nw = open(NW_data_file,'r')
doc_oe = open(OE_data_file,'r')
data = yml.load(doc_nw,Loader)
content_oe = doc_oe.readlines()

#Initialize Variables
map_type = str('jordan_wigner')
n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
print('repulsion: ',nuclear_repulsion_energy)
n_orbitals = 2*n_spatial_orbitals
n_particles = data['integral_sets'][0]['n_electrons']
# n_alpha = int(n_particles/2)
# n_beta = int(n_particles/2)
n_alpha = 3
n_beta = 4

#Populating the QMolecule class with the data to make calculations easier
qm.num_orbitals = n_spatial_orbitals
qm.num_alpha = n_alpha
qm.num_beta = n_beta
#This is just a place holder for freezing the core that I need for the MP2 calculations
qm.core_orbitals = 0
qm.nuclear_repulsion_energy = nuclear_repulsion_energy
hf_energy = data['integral_sets'][0]['scf_energy']['value']
qm.hf_energy = hf_energy

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

one_electron_spatial_integrals, two_electron_spatial_integrals = lh.get_spatial_integrals_noperm(one_electron_import,two_electron_import,n_spatial_orbitals)
qm.mo_eri_ints = two_electron_spatial_integrals
h1, h2 = lh.convert_to_spin_index(one_electron_spatial_integrals,two_electron_spatial_integrals, n_spatial_orbitals)

print('Getting Pauli Operator')
fop = FermionicOperator(h1, h2)
qop_paulis = fop.mapping(map_type)

start_time = time.time()

##############################################################################

# dist = 2*data['integral_sets'][0]['geometry']['atoms'][1]['coords'][2]


###########################################################
init_state = HartreeFock(n_orbitals, num_particles=[n_alpha, n_beta], qubit_mapping=map_type, two_qubit_reduction=False)

var_op = UCCSD(num_orbitals=n_orbitals, num_particles=[n_alpha, n_beta], reps=1, initial_state=init_state, qubit_mapping=map_type, \
               two_qubit_reduction=False, mp2_reduction=False, method_singles='both', method_doubles='ucc', excitation_type='sd')


print('There are {} params in this anzats'.format(var_op.num_parameters))
print(var_op._double_excitations)
dumpy_params = np.random.rand(var_op.num_parameters)
var_cirq = var_op.construct_circuit(dumpy_params)
print('Depth: ', var_cirq.depth())
print('size: ',var_cirq.size())
#print(var_cirq)
# print(var_cirq)
# qasm_cirq = var_cirq.qasm()
#print(qasm_cirq)

# Fout = open('UCCSpD_QASM_HF_Orb-{}_Dist-{}.qasm'.format(n_spatial_orbitals,dist),"w")
# Fout.write(qasm_cirq)
# Fout.close()