from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD

from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit import Aer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType

import numpy as np

mapping = ['jordan_wigner', 'bravyi_kitaev', 'parity']
geometry = 'H 0.0 0.0 {0}; H 0.0 0.0 -{0}'
dist = 0.5
basis = 'sto-3g'
driver = PySCFDriver(atom=geometry.format(dist), unit=UnitsType.ANGSTROM, charge=0, spin=0, basis=basis)
molecule = driver.run()


n_orbitals = molecule.num_orbitals
n_electrons = molecule.num_alpha+molecule.num_beta
n_qubits = 2*n_orbitals

one_electron_integrals = molecule.mo_onee_ints
two_electron_integrals = molecule.mo_eri_ints

h1 = molecule.onee_to_spin(one_electron_integrals)
h2 = molecule.twoe_to_spin(two_electron_integrals)

fop = FermionicOperator(h1, h2)
qop_paulis = fop.mapping(mapping[0])

active_occ = n_electrons//2
active_virt = (2*n_orbitals-n_electrons)//2
active_occ_list = np.arange(active_occ)
active_virt_list = np.arange(active_virt)
'''
Create the circuit for the UCCSD ansatz with initial Hartree-Fock state
'''

init_state = HartreeFock(num_qubits=n_qubits, num_orbitals=2*n_orbitals, num_particles=n_electrons,
                         qubit_mapping=mapping[0],two_qubit_reduction=False)
var_op = UCCSD(num_qubits=n_qubits, depth=1, num_orbitals=2*n_orbitals, num_particles=n_electrons, active_occupied=active_occ_list,\
               active_unoccupied=active_virt_list,initial_state=init_state, qubit_mapping=mapping[0], \
               two_qubit_reduction=False)

# The optimal params should calculate the GS wavefuntion with UCCSD
optimal_params = [-7.87546205e-06,  8.71366455e-06, -1.76217519e-01]
dumpy_params = np.random.rand(var_op.num_parameters)
var_cirq = var_op.construct_circuit(optimal_params)
print(var_cirq)





'''
There are two methods to get the energy and groundstate wavefunction and energy
calculated below. The first is the energy from hamiltonian diagonalization. 
The second method is the VQE algorithm. This is how the optimal parameters were
calculated. 

print('Getting energy')
exact_eigensolver = ExactEigensolver(qop_paulis, k=1)
ret = exact_eigensolver.run()
print('The electronic energy is: {:.12f}'.format(ret['eigvals'][0].real))

max_eval = 500
optimizer = COBYLA(maxiter=max_eval, disp=True, tol=1e-5, rhobeg=1e-1)
print('Doing VQE')
algorithm = VQE(qop_paulis,var_op,optimizer, initial_point=np.zeros(var_op.num_parameters))
result = algorithm.run(quantum_instance=quantum_instance)
vqe_energy = result['energy']
gs = result['eigvecs']
print(gs)
print('The VQE energy is: ',vqe_energy)
opt_params = result['opt_params']
print(' {} are the optimal parameters.'.format(opt_params))
'''



