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

import numpy as np
from scipy import linalg as la
import yaml as yml
from yaml import SafeLoader as Loader
import Load_Hamiltonians as lh
import os
import time
from datetime import date

#################
backend1 = Aer.get_backend('statevector_simulator')
backend2 = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend=backend1)
# #################

#Importing data generated from NW Chem to run experiments
#Li2_cc-pVTZ_4_ORBITALS_Li2-4_ORBITALS-ccpVTZ-2_1384
root_dir = '/Users/mmetcalf/Codes/Downfolding/Qiskit/'
NW_data_file = str(root_dir+'Li2-4_ORBITALS-ccpVTZ-2_673.yaml')
OE_data_file = str(root_dir+ 'Li2-4_ORBITALS-ccpVTZ-2_673.FOCK')
doc_nw = open(NW_data_file,'r')
doc_oe = open(OE_data_file,'r')
data = yml.load(doc_nw,Loader)
content_oe = doc_oe.readlines()

###########################################################
#Initialize Variables
map_type = str('parity')
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

one_electron_spatial_integrals, two_electron_spatial_integrals = lh.get_spatial_integrals(one_electron_import,two_electron_import,n_spatial_orbitals)

qm.mo_eri_ints = two_electron_spatial_integrals
# qm.orbital_energies = molecule.orbital_energies[:n_spatial_orbitals]

h1, h2 = lh.convert_to_spin_index(one_electron_spatial_integrals, two_electron_spatial_integrals, n_orbitals)
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
               two_qubit_reduction=two_qubit_reduction, mp2_reduction=True, singles_deletion=True)

print('There are {} params in this anzats'.format(var_op.num_parameters))

# setup a classical optimizer for VQE
max_eval = 3000
# Would using a different optimizer yield better results at long distances?
optimizer = COBYLA(maxiter=max_eval, disp=True, tol=1e-5, rhobeg=1e-1)

# Call the VQE algorithm class with your qubit operator for the Hamiltonian and the variational op
print('Doing VQE')
algorithm = VQE(qop_paulis,var_op,optimizer,'paulis', initial_point=var_op._mp2_coeff)

result = algorithm.run(quantum_instance=quantum_instance)
vqe_energy = result['energy']+nuclear_repulsion_energy
print('The VQE energy is: ',vqe_energy)
print('HartreeFock: ', hf_energy)
opt_params = result['opt_params']
print(' {} are the optimal parameters.'.format(opt_params))

end_time = time.time()
print('It took {} seconds to run this calculation'.format(end_time-start_time))

#Output Files to Plot stuff
Fout = open('VQEEnergy_HF_Orb-{}_Dist-{}_DUCC_{}.dat'.format(n_spatial_orbitals,dist, date.today()),"w")
my_info = [dist, vqe_energy]
my_info_to_str = " ".join(str(e) for e in my_info)
Fout.write(my_info_to_str + "\n")
Fout.close()

