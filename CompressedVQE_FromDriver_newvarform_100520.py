import yaml as yml
import numpy as np
from qiskit import Aer
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.chemistry.components.initial_states import HartreeFock
from new_var_form import NVARFORM
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.aqua.operators import WeightedPauliOperator
from yaml import SafeLoader as Loader
import os
import scipy.linalg as la
from qiskit.chemistry import QMolecule as qm
from qiskit.chemistry import FermionicOperator as fo
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.quantum_info.operators import pauli as p
import reduced_ham_qiskit as rh
from qiskit.aqua import operators
from itertools import product

import Load_Hamiltonians as lh
from Basis import Basis


Fout = open('H2_VQEEnergies_wMP2_WSingles_7-orbitals_091119.dat',"w")
Fout_op = open('H2_OptimalParams_wMP2_7-orbitals_091119.dat',"w")
#Fout = open('Li2_ExactEnergiesG&FE_wMP2_4-orbitals_052919.dat',"w")
###########################################################

output_data = []
#Looping over all of the yaml files in a particular directory and saving the energies from VQE and Exact
ind = 0

distances = [0.7]
# geometry = 'H 0.0 0.0 -{0}; H 0.0 0.0 {0}'
gmetry = ' C 0.0000 0.0000 0.0000; C 1.4000 0.0000 0.0000; C 2.1000 1.2124 0.0000; C 1.4000 2.4249 0.0000; C 0.0000 2.4249 0.0000; C -0.7000 1.2124 0.0000; H -0.5500 -0.9526 0.0000; H -0.5500 3.3775 0.0000; H 1.9500 -0.9526 0.0000; H -1.8000 1.2124 0.0000; H 3.2000 1.2124 0.0000; H 1.9500 3.3775 0.0000'
# Maybe try this to get the benzene molecule
# http://openmopac.net/manual/symmetry_output.html
basis = 'STO-6G'

for key, dist in enumerate(distances):

    ############### RUN PYSCF DRIVER ################
    print('{} is the distance'.format(dist))
    # driver = PySCFDriver(atom=geometry.format(dist), unit=UnitsType.ANGSTROM, basis=basis)
    driver = PySCFDriver(atom=geometry, unit=UnitsType.ANGSTROM, basis=basis)
    molecule = driver.run()

    ################### INITIALIZE VARIABLES #################
    # n_spatial_orbitals = molecule.num_orbitals
    n_spatial_orbitals = 6
    molecule.num_orbitals = n_spatial_orbitals
    print('There are {} orbitals.'.format(n_spatial_orbitals))
    print('This is the core',molecule.core_orbitals)
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
    n_orbitals = 2*n_spatial_orbitals
    n_particles = molecule.num_alpha + molecule.num_beta
    print('There are {} particles'.format(n_particles))
    qm.hf_energy = molecule.hf_energy
    qm.orbital_energies = molecule.orbital_energies[:n_spatial_orbitals]
    qm.nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
    qm.num_alpha = molecule.num_alpha
    qm.num_beta = molecule.num_beta
    qm.num_orbitals = n_spatial_orbitals
    # fermion_basis = Basis(n_spatial_orbitals, molecule.num_alpha, molecule.num_beta, basis_type='block')
    fermion_basis = Basis(n_spatial_orbitals, 3, 3, basis_type='block')
    reduced_basis = fermion_basis.create_total_basis()
    n_qubits = fermion_basis._num_qubits
    print('There are {} qubits'.format(n_qubits))
    single_excitations, double_excitations = fermion_basis.compute_varform_excitations(reduced_basis)
    ################### CONSTRUCT HAMILTONIAN ####################
    one_electron_integrals = molecule.mo_onee_ints[:n_spatial_orbitals,:n_spatial_orbitals]

    two_electron_integrals = molecule.mo_eri_ints[:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals]

    qm.mo_eri_ints = two_electron_integrals

    h1 = qm.onee_to_spin(one_electron_integrals)
    h2 = qm.twoe_to_spin(np.einsum('ijkl->ljik', two_electron_integrals))
    print('have integrals')
    ################# IBM BACKEND #####################
    backend1 = Aer.get_backend('statevector_simulator',)
    backend2 = Aer.get_backend('qasm_simulator')


    # #################change indices because OF uses physicist notation #################################


    molecular_hamiltonian = fo(h1, h2)
    print('Mapping the hamiltonian')
    # Build matrix representation & diagonalize
    operator = molecular_hamiltonian.mapping("jordan_wigner", threshold=1e-1000)


    indices = []
    for key, val in enumerate(reduced_basis):
        indices.append(val)
    print('qubit index: ',indices)
    print(len(indices))
    qubit_hamiltonian_matrix = operators.op_converter.to_matrix_operator_reduced(operator, indices, n_qubits).dense_matrix
    print(np.shape(qubit_hamiltonian_matrix))
    qubit_hamiltonian_matrix2 = operators.op_converter.to_matrix_operator(operator).dense_matrix
    evltemp2, evctemp2 = la.eigh(qubit_hamiltonian_matrix)
    evltemp1, evctemp1 = la.eigh(qubit_hamiltonian_matrix2)
    print("reduced hamiltonian eigenvalues: ", evltemp2)
    print("reg hamiltonian eigenvalues: ", evltemp1)

    'Generate Pauli string acting on qubit space'
    input_array_lst = []
    pauli_basis = []

    for i in product(range(2), repeat=n_qubits):
        input_array_lst.append(np.array(i))


    for i in range(len(input_array_lst)):
        for j in range(len(input_array_lst)):
            pauli_basis.append(p.Pauli(input_array_lst[i], input_array_lst[j]))


    "Get coefficients"
    coefficients = {}
    nonzero_pauli_lst_matrix = []
    nonzero_pauli_lst = []
    nonzero_pauli = []
    count = 0
    for pauli in pauli_basis:
        norm = np.sqrt(np.sum(np.square(pauli.to_matrix())))
        if norm is complex:
            ham_pauli_prod = np.dot(-1j * qubit_hamiltonian_matrix, pauli.to_matrix())
        else:
            ham_pauli_prod = np.dot(qubit_hamiltonian_matrix, pauli.to_matrix())
        trace = np.trace(ham_pauli_prod)
        if trace != complex(0, 0):
            nonzero_pauli_lst_matrix.append(pauli.to_matrix())
            nonzero_pauli_lst.append(pauli.to_label())
            norm = np.sqrt(np.sum(np.square(pauli.to_matrix())))
            # nonzero_pauli.append([trace/(n_qubits**2), pauli])
            nonzero_pauli.append([trace / (2**n_qubits), pauli])
        coefficients.update({pauli.to_label(): trace/(2**n_qubits)})

    norm_lst = []
    print(pauli_basis)
    for pauli in nonzero_pauli_lst_matrix:
        norm_lst.append(np.sqrt(np.sum(np.square(pauli))))

    print(norm_lst)
    print('The number of terms in original pauli basis: \n', len(pauli_basis))


    print(nonzero_pauli_lst)
    print(coefficients)
    print("The number of terms in the Pauli basis: \n", len(nonzero_pauli_lst))
    test_matrix = nonzero_pauli_lst_matrix[0] * coefficients.get(nonzero_pauli_lst[0])
    count = 1
    for i in nonzero_pauli_lst_matrix[1:]:
        test_matrix += (i * coefficients.get(nonzero_pauli_lst[count]))
        count += 1

    print(np.shape(test_matrix))
    eigen_val_init, eigen_vec_init = la.eigh(qubit_hamiltonian_matrix)
    print("\nThe eigenvalues for the initial 256x256 Hamiltonian:")
    print(eigen_val_init)

    print("\nEigenvalues of reduced Hamiltonian: ")
    eigen_val, eigen_vec = la.eigh(test_matrix)
    print(eigen_val)

    print('There are {} Paulis'.format(len(nonzero_pauli)))
    map_type = str('jordan_wigner')
    qop_pauli = WeightedPauliOperator(nonzero_pauli)
    print("nonzero_pauli and type ", nonzero_pauli, type(nonzero_pauli))

    fo._convert_to_interleaved_spins(molecular_hamiltonian)
    qop_paulis2 = molecular_hamiltonian.mapping(map_type=map_type)
    qop2 = WeightedPauliOperator(qop_paulis2.paulis)

    #print("this should be the non reduced operator paulis ", qop2)

    #Get Variational form and intial state
    single_excitations, double_excitations = fermion_basis.compute_varform_excitations(reduced_basis)
    var_op = NVARFORM(num_qubits=n_qubits, depth=1, num_orbitals=n_orbitals//2, num_particles=n_particles, excitation_list=[single_excitations,double_excitations], mp2_reduction = True)


    ###################### VQE RESULT ###############################
    setup a classical optimizer for VQE
    max_eval = 500
    optimizer = COBYLA(maxiter=max_eval, disp=True, rhobeg=1e-1, tol=1e-4)


    initial_params = 1e-8 * np.random.rand(var_op._num_parameters)
    print(n_particles, n_orbitals)
    #print('operators: \n', qop_pauli, '\n', var_op)
    # algorithm = VQE(qop_paulis2, var_op, optimizer, initial_point=initial_params)

    algorithm = VQE(qop_pauli, var_op, optimizer, initial_point=initial_params, include_custom=True)

    result = algorithm.run(backend2)
    vqe_energy = result['energy'] + nuclear_repulsion_energy
    vqe_params = result['optimal_parameters']

    print('Have the VQE energy')
    qop = WeightedPauliOperator(paulis=qop_pauli.paulis)

    exact_eigensolver = ExactEigensolver(qop, k=1)
    ret = exact_eigensolver.run()
    print('The electronic energy is: {:.12f}'.format(ret['eigvals'][0].real))
    print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real + nuclear_repulsion_energy))
    exact_energy = ret['eigvals'][0].real + nuclear_repulsion_energy
    print(nuclear_repulsion_energy)
    print("exact: ", exact_energy, "\n vqe: ", vqe_energy)
    print(vqe_params)

    ###################################################################
    # The outputs to plot
    my_info = [dist, exact_energy, vqe_energy]
    param_info = [dist, vqe_params]
    output_data.append(my_info)
    my_info_to_str = "\t".join(str(e) for e in my_info)
    params_to_str = "\t".join(str(e) for e in param_info)
    Fout.write(my_info_to_str + "\n")
    Fout_op.write(params_to_str + "\n")

print(output_data)
Fout.close()
Fout_op.close()