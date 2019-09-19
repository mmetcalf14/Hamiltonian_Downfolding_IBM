from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry import QMolecule
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.aqua.operators import Z2Symmetries

from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua import Operator
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit import Aer
from qiskit.quantum_info import Pauli

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import numpy as np

import time

def symmetries_4_8qbit():

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

def trunctate_spatial_integrals(one_electron, two_electron, trunc):
    one_electron = np.where(
        (abs(one_electron) < trunc), 0,
        one_electron)
    two_electron = np.where(
        (abs(two_electron) < trunc), 0,
        two_electron)

    return one_electron,two_electron

#################
backend1 = Aer.get_backend('statevector_simulator')
backend2 = Aer.get_backend('qasm_simulator')

###########################################################
# Output Files to Plot stuff

Fout = open('Energy_H2_Orb-4_cc-pvtz_noSingles_wMP2_Jcurve2_082919.dat', "w")
###########################################################
#Initialize Variables
map_type = str('jordan_wigner')
truncation_threshold = 0.0001

geometry = 'Li 0.0 0.0 0.0; H 0.0 0.0 {0}'
# geometry = 'H 0.0 0.0 -{0}; H 0.0 0.0 {0}'
basis = 'cc-pvtz'

# li2_distances = [1.0692, 1.136025, 1.20285, 1.269675, 1.3365, 1.403325,
#                  1.536975, 1.670625, 1.8, 1.9, 2.00475, 2.1, 2.338875, 2.4, 2.5,
#                  2.68, 3.0, 3.2, 3.425, 4., 5.]
# li2_distances = [2.9403/2., 3.2076/2., 6.6825/2.]
# H2_dist = [0.7004]
LiH_dist = [1.5]

for key, dist in enumerate(LiH_dist):

    ############### RUN PYSCF DRIVER ################
    print('{} is the distance'.format(dist))
    driver = PySCFDriver(atom=geometry.format(dist), unit=UnitsType.ANGSTROM, charge=0, spin=0, basis=basis)
    molecule = driver.run()
    print('{} orbitals'.format(molecule.num_orbitals))
    orb_deletion = [i for i in range(4, molecule.num_orbitals)]

    print('Instantiating Hamiltonian')
    hamiltonian = Hamiltonian(transformation=TransformationType.FULL,qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                              two_qubit_reduction=True,
                              freeze_core=False, orbital_reduction=orb_deletion)
    print('Getting qubit operator')
    qop_paulis, _ = hamiltonian.run(molecule)

    ############## Z2 Symmetry Reduction #############

    # if map_type == 'parity':
    #     two_qubit_reduction = True
    #     n_qubits = n_orbitals - 2
    #     # qop_paulis = Z2Symmetries.two_qubit_reduction(qop_paulis, n_particles)
    # else:
    #     two_qubit_reduction = False
    #     n_qubits = n_orbitals
    #
    # z2_symmetries = Z2Symmetries.find_Z2_symmetries(qop_paulis)
    symm, sq_pauli, sq_list = symmetries_4_8qbit()
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
    # ################## EXACT ENERGY ##################
    #
    # print('Getting energy')
    # exact_eigensolver = ExactEigensolver(qop_paulis, k=1)
    # ret = exact_eigensolver.run()
    # print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real))
    # exact_energy = ret['eigvals'][0].real + molecule.nuclear_repulsion_energy
    #
    #
    # ##################### RUN VQE ######################
    #
    # start_time = time.time()
    #
    # init_state = HartreeFock(n_qubits, n_orbitals, n_particles, map_type,two_qubit_reduction=two_qubit_reduction)
    #
    # var_op = UCCSD(num_qubits=n_qubits, depth=1, num_orbitals=n_orbitals, num_particles=n_particles, active_occupied=active_occ_list,\
    #                active_unoccupied=active_virt_list,initial_state=init_state, qubit_mapping=map_type,
    #                two_qubit_reduction=two_qubit_reduction, mp2_reduction=True, singles_deletion=True)
    #
    # # setup a classical optimizer for VQE
    # max_eval = 1000
    # optimizer = COBYLA(maxiter=max_eval,disp=True, tol=1e-4)
    # print('params: ',var_op.num_parameters)
    #
    # # #Call the VQE algorithm class with your qubit operator for the Hamiltonian and the variational op
    # print('Doing VQE')
    # algorithm = VQE(qop_paulis,var_op,optimizer,'paulis', initial_point=None)
    # result = algorithm.run(backend1)
    # vqe_energy = result['energy']+nuclear_repulsion_energy
    # print('The VQE energy is: ',result['energy']+nuclear_repulsion_energy)
    # end_time = time.time()
    # print('It took {} seconds to run this calculation'.format(end_time-start_time))
    #
    # ################### OUTPUT ENERGIES ####################
    # my_info = [2*dist, exact_energy, vqe_energy, molecule.hf_energy]
    # my_info_to_str = "\t".join(str(e) for e in my_info)
    #
    # Fout.write(my_info_to_str + "\n")

Fout.close()







