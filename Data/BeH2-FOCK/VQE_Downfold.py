from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry import QMolecule as qm
from qiskit.aqua.components.optimizers import COBYLA, L_BFGS_B
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.operators.expectations import PauliExpectation, AerPauliExpectation
from qiskit import Aer, BasicAer

import numpy as np
import yaml as yml
from yaml import SafeLoader as Loader
import Load_Hamiltonians as lh
import time

#################
backend1 = Aer.get_backend('statevector_simulator')
backend2 = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(shots=2**10, backend=backend2)
#################

#Importing data generated from NW Chem to run experiments
#root_dir = '~/dir'
#NW_data_file = str(root_dir+'Li2-10_ORBITALS-ccpVTZ-2_673.yaml')
#OE_data_file = str(root_dir+ 'Li2-NO-2_673.FOCK')
NW_data_file = str('Nat_Orb/BeH2-I.yaml')
OE_data_file = str('Nat_Orb/BeH2-NO-DUCC-I')
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
n_alpha = int(n_particles/2)
n_beta = int(n_particles/2)

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

###########################################
#Exact Result to compare to: This can also be obtained via the yaml file once we verify correctness.
#Don't use trunctated Hamiltonian
# print('Getting energy')
# exact_eigensolver = ExactEigensolver(qop_paulis, k=1)
# ret = exact_eigensolver.run()
# print('The electronic energy is: {:.12f}'.format(ret['eigvals'][0].real))
# print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real + nuclear_repulsion_energy))
###########################################

init_state = HartreeFock(n_orbitals, num_particles=[n_alpha, n_beta], qubit_mapping=map_type, two_qubit_reduction=False)

var_op = UCCSD(num_orbitals=n_orbitals, num_particles=[n_alpha, n_beta], reps=1, initial_state=init_state, qubit_mapping=map_type, \
               two_qubit_reduction=False, mp2_reduction=True, method_singles='both', method_doubles='ucc', excitation_type='sd')

print('There are {} params in this anzats'.format(var_op.num_parameters))
start_time = time.time()

max_eval = 10000
optimizer = COBYLA(maxiter=max_eval, disp=True, tol=1e-3, rhobeg=1e-1)
print('Doing VQE')
print(var_op._mp2_coeff)
initial_point = np.zeros(len(var_op._single_excitations))
initial_point = np.append(initial_point,var_op._mp2_coeff)

algorithm = VQE(operator=qop_paulis,var_form=var_op, optimizer=optimizer, initial_point=initial_point, include_custom=True)
result = algorithm.run(backend2)
vqe_energy = result['eigenvalue']+nuclear_repulsion_energy
print('The VQE energy is: ',vqe_energy)
print('HartreeFock: ', hf_energy)
opt_params = result['optimal_point']
print(' {} are the optimal parameters.'.format(opt_params))

end_time = time.time()
print('It took {} seconds to run this calculation'.format(end_time-start_time))
