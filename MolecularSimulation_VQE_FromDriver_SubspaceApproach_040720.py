from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD #Our UCCSD
from qiskit.chemistry import QMolecule as qm
from qiskit.aqua.operators import Z2Symmetries

from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua import Operator
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit import Aer

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import numpy as np
# import Basis
import time


Fout = open('Energy_H2_Orb-4_cc-pvtz_noSingles_wMP2_Jcurve2_082919.dat', "w")
geometry = 'H 0.0 0.0 -{0}; H 0.0 0.0 {0}'
basis = 'cc-pvtz'
map_type = 'jordan_wigner'
H2_dist = np.arange(0,5,.1)

for key, dist in enumerate(H2_dist):

    ############### RUN PYSCF DRIVER ################
    print('{} is the distance'.format(dist))
    driver = PySCFDriver(atom=geometry.format(dist), unit=UnitsType.ANGSTROM, basis=basis)
    molecule = driver.run()

    ################### INITIALIZE VARIABLES #################
    n_spatial_orbitals = molecule.num_orbitals
    print(' {} orbitals space in the {} basis'.format(n_spatial_orbitals,basis))
    # Reduce orbital space to 4 orbitals
    n_spatial_orbitals = 4
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy
    print('repulsion: ',nuclear_repulsion_energy)
    n_orbitals = 2*n_spatial_orbitals
    n_spin_up = molecule.num_alpha
    n_spin_down = molecule.num_beta
    n_particles = molecule.num_alpha + molecule.num_beta
    molecule.orbital_energies = molecule.orbital_energies[:n_spatial_orbitals]

    # Call basis class and get total basis for qubit mapping

    one_electron_integrals = molecule.mo_onee_ints[:n_spatial_orbitals,:n_spatial_orbitals]
    two_electron_integrals = molecule.mo_eri_ints[:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals]
    qm.mo_eri_ints = two_electron_integrals

    h1 = qm.onee_to_spin(one_electron_integrals)
    h2 = qm.twoe_to_spin(np.einsum('ijkl->ljik', two_electron_integrals))

    fop = FermionicOperator(h1, h2)
    # In OpenFermion can the fermion operator be converted into a matrix?
    qop_paulis = fop.mapping(map_type)

    # Convert Hamiltonian into new basis here.

    # Get new UCCSD operator from our class

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