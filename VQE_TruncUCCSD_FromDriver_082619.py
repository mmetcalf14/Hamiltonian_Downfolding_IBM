from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry import QMolecule as qm

from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua import Operator
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit import Aer

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import numpy as np

import time

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

Fout = open('Energy_Li2_Orb-4_cc-pvtz_noSingles_wMP2_Jcurve2_082919.dat', "w")
###########################################################
#Initialize Variables
map_type = str('jordan_wigner')
truncation_threshold = 0.0001

geometry = 'Li 0.0 0.0 -{0}; Li 0.0 0.0 {0}'
basis = 'cc-pvtz'

# li2_distances = [1.0692, 1.136025, 1.20285, 1.269675, 1.3365, 1.403325,
#                  1.536975, 1.670625, 1.8, 1.9, 2.00475, 2.1, 2.338875, 2.4, 2.5,
#                  2.68, 3.0, 3.2, 3.425, 4., 5.]
li2_distances = [2.9403/2., 3.2076/2., 6.6825/2.]

for key, dist in enumerate(li2_distances):

    ############### RUN PYSCF DRIVER ################
    print('{} is the distance'.format(dist))
    driver = PySCFDriver(atom=geometry.format(dist), unit=UnitsType.ANGSTROM, basis=basis)
    molecule = driver.run()

    ################### INITIALIZE VARIABLES #################
    n_spatial_orbitals = 6
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
    print('repulsion: ',nuclear_repulsion_energy)
    n_orbitals = 2*n_spatial_orbitals
    n_particles = molecule.num_alpha + molecule.num_beta

    if map_type == 'parity':
        #For two-qubit reduction
        n_qubits = n_orbitals-2
    else:
        n_qubits = n_orbitals

    #Populating the QMolecule class with the data to make calculations easier
    qm.num_orbitals = n_spatial_orbitals
    qm.num_alpha = molecule.num_alpha
    qm.num_beta = molecule.num_beta
    #This is just a place holder for freezing the core that I need for the MP2 calculations
    qm.core_orbitals = 0
    qm.nuclear_repulsion_energy = nuclear_repulsion_energy
    qm.hf_energy = molecule.hf_energy
    qm.orbital_energies = molecule.orbital_energies[:n_spatial_orbitals]

    active_occ = n_particles//2
    active_virt = (n_orbitals-n_particles)//2
    active_occ_list = np.arange(active_occ)
    active_occ_list = active_occ_list.tolist()
    active_virt_list = np.arange(active_virt)
    active_virt_list = active_virt_list.tolist()

    ################### CONSTRUCT HAMILTONIAN ####################

    one_electron_integrals = molecule.mo_onee_ints[:n_spatial_orbitals,:n_spatial_orbitals]

    two_electron_integrals = molecule.mo_eri_ints[:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals]
    one_electron_integrals, two_electron_integrals = trunctate_spatial_integrals(one_electron_integrals,two_electron_integrals, truncation_threshold)

    qm.mo_eri_ints = two_electron_integrals

    h1 = qm.onee_to_spin(one_electron_integrals)
    h2 = qm.twoe_to_spin(np.einsum('ijkl->ljik', two_electron_integrals))

    fop = FermionicOperator(h1, h2)
    qop_paulis = fop.mapping(map_type)
    qop = Operator(paulis=qop_paulis.paulis)

    ################## EXACT ENERGY ##################

    # print('Getting energy')
    exact_eigensolver = ExactEigensolver(qop, k=1)
    ret = exact_eigensolver.run()
    print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real + nuclear_repulsion_energy))
    exact_energy = ret['eigvals'][0].real + nuclear_repulsion_energy


    ##################### RUN VQE ######################

    start_time = time.time()

    init_state = HartreeFock(n_qubits, n_orbitals, n_particles, map_type,two_qubit_reduction=False)
    var_op = UCCSD(num_qubits=n_qubits, depth=1, num_orbitals=n_orbitals, num_particles=n_particles, active_occupied=active_occ_list,\
                   active_unoccupied=active_virt_list,initial_state=init_state, qubit_mapping=map_type, mp2_reduction=True, singles_deletion=True)

    # setup a classical optimizer for VQE
    max_eval = 1000
    optimizer = COBYLA(maxiter=max_eval,disp=True, tol=1e-4)
    print('params: ',var_op.num_parameters)

    # #Call the VQE algorithm class with your qubit operator for the Hamiltonian and the variational op
    print('Doing VQE')
    algorithm = VQE(qop_paulis,var_op,optimizer,'paulis', initial_point=None)
    result = algorithm.run(backend1)
    vqe_energy = result['energy']+nuclear_repulsion_energy
    print('The VQE energy is: ',result['energy']+nuclear_repulsion_energy)
    end_time = time.time()
    print('It took {} seconds to run this calculation'.format(end_time-start_time))

    ################### OUTPUT ENERGIES ####################
    my_info = [2*dist, exact_energy, vqe_energy, molecule.hf_energy]
    my_info_to_str = "\t".join(str(e) for e in my_info)

    Fout.write(my_info_to_str + "\n")

Fout.close()







