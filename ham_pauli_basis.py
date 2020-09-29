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
from qiskit.quantum_info.operators import pauli as p
import reduced_ham_qiskit as rh
from qiskit.aqua import operators
from itertools import product

import Load_Hamiltonians as lh

################### WALK ROOT DIR ############################

root_dir = '/Users/dchamaki/Hamiltonian_Downfolding_IBM/IntegralData/H2_MEKENA'

data_file_list = []
data_file_list_oe = []
for dirName, subdirList, fileList in os.walk(root_dir):
    print('Found directory: %s' % dirName)
    for fname in sorted(fileList):
        if fname.endswith('.yaml'):
            data_file_list.append(fname)
        if fname.endswith('.FOCK'):
            data_file_list_oe.append(fname)

#
data_file_list.remove('h2_ccpvtz_ccsd_1_4008au_ducc_1_3.yaml')
data_file_list.remove('h2_ccpvtz_ccsd_4_00au_ducc_1_3.yaml')
data_file_list.remove('h2_ccpvtz_ccsd_10_0au_ducc_1_3.yaml')
# data_file_list.remove(('h2_ccpvtz_ccsd_0_80au_ducc_1_3.yaml'))
#
# data_file_list_oe.remove('h2_ccpvtz_ccsd_0_80au.FOCK')
data_file_list_oe.remove('h2_ccpvtz_ccsd_1_4008au.FOCK')
data_file_list_oe.remove('h2_ccpvtz_ccsd_10_0au.FOCK')
data_file_list_oe.remove(('h2_ccpvtz_ccsd_4_00au.FOCK'))

############# Output Files to Plot stuff ###############
print(data_file_list)
Fout = open('H2_VQEEnergies_wMP2_WSingles_7-orbitals_091119.dat',"w")
Fout_op = open('H2_OptimalParams_wMP2_7-orbitals_091119.dat',"w")
#Fout = open('Li2_ExactEnergiesG&FE_wMP2_4-orbitals_052919.dat',"w")
###########################################################

output_data = []
#Looping over all of the yaml files in a particular directory and saving the energies from VQE and Exact
ind = 0

for file1, file2 in zip(data_file_list, data_file_list_oe):

    NW_data_file = str(os.path.join(root_dir,file1))

    OE_data_file = str(os.path.join(root_dir,file2))

    try:
        doc = open(NW_data_file, 'r')
        doc_oe = open(OE_data_file,'r')
        data = yml.load(doc, Loader)
        content_oe = doc_oe.readlines()
    finally:
        doc.close()
        doc_oe.close()

    # Import all the data from a yaml file
    n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
    nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
    n_orbitals = 2 * n_spatial_orbitals
    n_particles = data['integral_sets'][0]['n_electrons']
    coordinates1 = data['integral_sets'][0]['geometry']['atoms'][0]['coords']
    coordinates2 = data['integral_sets'][0]['geometry']['atoms'][1]['coords']
    dist = np.sqrt((coordinates2[0] - coordinates1[0]) ** 2 + (coordinates2[1] - coordinates1[1]) ** 2 + (
                coordinates2[2] - coordinates1[2]) ** 2)
    # R.append(round(dist,3))
    print('orbitals: ', n_orbitals)
    print(n_particles)
    print('Bond distance is {}'.format(dist))
    n_qubits = 4 #forgot how to calculate reduced number of qubits formualically, ask!!

    # Importing the integrals
    truncation_threshold = 1e-1000

    one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
    two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

    # Getting spatial integrals and spin integrals to construct Hamiltonian
    one_electron_spatial_integrals, two_electron_spatial_integrals = lh.get_spatial_integrals(one_electron_import,
                                                                                              two_electron_import,
                                                                                              n_spatial_orbitals)
    one_electron_spatial_integrals, two_electron_spatial_integrals = lh.trunctate_spatial_integrals(
        one_electron_spatial_integrals, two_electron_spatial_integrals, .001)
    h1, h2 = lh.convert_to_spin_index(one_electron_spatial_integrals, two_electron_spatial_integrals,
                                      n_spatial_orbitals, truncation_threshold)

    # print(h2)

    ################# IBM BACKEND #####################
    backend1 = Aer.get_backend('statevector_simulator',)
    backend2 = Aer.get_backend('qasm_simulator')


    # #################change indices because OF uses physicist notation #################################
    two_electron_spatial_integrals = np.einsum('ijkl->iklj', two_electron_spatial_integrals)
    qm.num_orbitals = n_spatial_orbitals
    qm.num_alpha = n_particles // 2
    qm.num_beta = n_particles // 2
    qm.core_orbitals = 0
    qm.nuclear_repulsion_energy = nuclear_repulsion_energy
    qm.hf_energy = data['integral_sets'][0]['scf_energy']['value']


    del [one_electron_spatial_integrals]
    del [two_electron_spatial_integrals]

    molecular_hamiltonian = fo(h1, h2)

    # Build matrix representation & diagonalize
    operator = molecular_hamiltonian.mapping("jordan_wigner", threshold=1e-1000)
    indices = [3, 6, 9, 12, 18, 24, 33, 36, 48, 66, 72, 96, 129, 132, 144, 192]
    qubit_hamiltonian_matrix = operators.op_converter.to_matrix_operator_reduced(operator, indices, n_qubits).dense_matrix
    qubit_hamiltonian_matrix2 = operators.op_converter.to_matrix_operator(operator).dense_matrix
    evltemp2, evctemp2 = la.eigh(qubit_hamiltonian_matrix)
    print("reduced hamiltonian eigenvalues: ", evltemp2)

    'Generate Pauli string acting on qubit space'
    input_array_lst = []
    pauli_basis = []

    for i in product(range(2), repeat=n_qubits):
        input_array_lst.append(np.array(i))

    print(input_array_lst)

    for i in range(len(input_array_lst)):
        for j in range(len(input_array_lst)):
            pauli_basis.append(p.Pauli(input_array_lst[i], input_array_lst[j]))

    print(pauli_basis)

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
            nonzero_pauli.append([trace/16, pauli])
        coefficients.update({pauli.to_label(): trace/16})

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

    eigen_val_init, eigen_vec_init = la.eigh(qubit_hamiltonian_matrix)
    print("\nThe eigenvalues for the initial 256x256 Hamiltonian:")
    print(eigen_val_init)

    print("\nEigenvalues of reduced Hamiltonian: ")
    eigen_val, eigen_vec = la.eigh(test_matrix)
    print(eigen_val)

    map_type = str('jordan_wigner')
    qop_pauli = WeightedPauliOperator(nonzero_pauli)
    print("nonzero_pauli and type ", nonzero_pauli, type(nonzero_pauli))

    fo._convert_to_interleaved_spins(molecular_hamiltonian)
    qop_paulis2 = molecular_hamiltonian.mapping(map_type=map_type)
    qop2 = WeightedPauliOperator(qop_paulis2.paulis)

    print("this should be the non reduced operator paulis ", qop2)

    #Get Variational form and intial state

    init_state = HartreeFock(n_orbitals, n_particles, map_type, two_qubit_reduction=False)
    var_op2 = UCCSD(num_orbitals=n_orbitals, num_particles=n_particles, initial_state=init_state, qubit_mapping=map_type, two_qubit_reduction=False)
    var_op = NVARFORM(num_qubits=4,depth=1, num_orbitals=n_orbitals//2, num_particles=n_particles, initial_state=init_state,
                qubit_mapping=map_type, two_qubit_reduction=False)

    print("uccsd: ", var_op2.parameter_bounds)
    print("our var form: ", var_op.parameter_bounds)

    ######################## VQE RESULT ###############################
    # setup a classical optimizer for VQE
    max_eval = 1
    optimizer = COBYLA(maxiter=max_eval, disp=True, tol=1e-4)


    initial_params = 1e-8 * np.random.rand(7)
    print(initial_params)
    optimal_params_0_8 = [0.01219819, 0.02912151, -0.01852697, 0.01863373, 0.0311612]
    optimal_params_1_4 = np.array([0.03909578, 0.03560041, -0.04395869, 0.04399395, 0.05516548])
    optimal_params_4 = np.array([0.48836229, 0.07245607, -0.07464593, 0.00588291, -0.121308, 0.12098167, 0.03228614])
    optimal_params_10 = np.array([0.7726071, 0.11747436, -0.12602609, -0.01833866, -0.11863041, 0.11783347, 0.01748658])

    print('Doing VQE')
    print(n_particles, n_orbitals)
    print('operators: \n', qop_pauli, '\n', var_op)
    # algorithm = VQE(qop_paulis2, var_op, optimizer, initial_point=initial_params)

    algorithm = VQE(qop_pauli, var_op, optimizer, initial_point=initial_params)
    # algorithm = VQE(qop_paulis2, var_op2, optimizer)


    result = algorithm.run(backend1)
    vqe_energy = result['energy'] + nuclear_repulsion_energy
    vqe_params = result['opt_params']

    qop = WeightedPauliOperator(paulis=qop_pauli.paulis)

    exact_eigensolver = ExactEigensolver(qop, k=1)
    ret = exact_eigensolver.run()
    print('The electronic energy is: {:.12f}'.format(ret['eigvals'][0].real))
    print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real + nuclear_repulsion_energy))
    exact_energy = ret['eigvals'][0].real + nuclear_repulsion_energy

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