from qiskit_chemistry import FermionicOperator, QMolecule
from qiskit_chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit_chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit_chemistry import QMolecule as qm

from qiskit_aqua.components.optimizers import COBYLA
from qiskit_aqua import Operator
from qiskit_aqua.algorithms import VQE, ExactEigensolver
from qiskit import Aer

from scipy import linalg as la
import numpy as np
import yaml as yml
from yaml import SafeLoader as Loader
import os

from dask.distributed import Client, LocalCluster
import dask


import Load_Hamiltonians as lh

#Define functions that can be used in parallel algorithm
def read_variables(root_dir, file1, file2):

    Ham_data_file = str(os.path.join(root_dir, file1))

    OrbE_data_file = str(os.path.join(root_dir + '/DOWNFOLDED_ORBITAL_ENERGIES', file2))

    try:
        doc_nw = open(Ham_data_file, 'r')
        doc_orbe = open(OrbE_data_file, 'r')
        dat = yml.load(doc_nw, Loader)
        content_orbe = doc_orbe.readlines()
    finally:
        doc_nw.close()
        doc_orbe.close()

    # Import all the data from a yaml file
    print('Getting data')

    #Importing the integrals
    one_e = dat['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
    two_e = dat['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

    num_spatial_orbitals = dat['integral_sets'][0]['n_orbitals']
    coulomb_energy = dat['integral_sets'][0]['coulomb_repulsion']['value']
    num_orbitals = 2 * num_spatial_orbitals
    num_particles = dat['integral_sets'][0]['n_electrons']
    d = 2 * dat['integral_sets'][0]['geometry']['atoms'][1]['coords'][2]
    print('Bond distance is {}'.format(d))
    if map_type == 'parity':
        # For two-qubit reduction
        num_qubits = num_orbitals - 2
    else:
        num_qubits = num_orbitals

    # Populating the QMolecule class with the data to make calculations easier - No need for returns
    # qm.num_orbitals = num_spatial_orbitals
    # qm.num_alpha = num_particles // 2
    # qm.num_beta = num_particles // 2
    # qm.core_orbitals = 0
    # qm.nuclear_repulsion_energy = coulomb_energy
    # qm.hf_energy = dat['integral_sets'][0]['scf_energy']['value']
    ###################Get Orbital Energies from FOCK file########################
    orbital_energies = []
    found = False
    count = 0
    for line in content_orbe:
        if 'Eigenvalues:' in line.split()[0]:
            found = True
        if found and len(line.split()) > 1:
            orbital_energies.append(float(line.split()[1]))
            count += 1
            if count >= num_spatial_orbitals:
                break
    qm.orbital_energies = orbital_energies

    return num_spatial_orbitals, num_orbitals, num_particles, num_qubits, d, coulomb_energy, one_e, two_e


def get_molecular_qubit_operator(one_ints, two_ints, n_orbitals, thresh, qubit_map):

    # Getting spatial integrals and spin integrals to construct Hamiltonian
    one_electron_spatial_integrals, two_electron_spatial_integrals = lh.get_spatial_integrals(one_ints,
                                                                                              two_ints,
                                                                                              n_orbitals)
    h1, h2 = lh.convert_to_spin_index(one_electron_spatial_integrals, two_electron_spatial_integrals,
                                                  n_orbitals, thresh)
    # For the MP2 Calculation
    qm.mo_eri_ints = two_electron_spatial_integrals

    # Constructing the fermion operator and qubit operator from integrals data
    fop = FermionicOperator(h1, h2)
    qop_paulis = fop.mapping(qubit_map)

    return qop_paulis


def run_vqe(pauli_operator, num_orbitals, num_particles, num_qubits, coulomb_energy, mapping, backend):

    var_energy = 0
    #Get Variational form and intial state
    initial_state = HartreeFock(num_qubits, num_orbitals, num_particles, mapping, two_qubit_reduction=False)
    var_op = UCCSD(num_qubits, 1, num_orbitals, num_particles, active_occupied=None, active_unoccupied=None, initial_state=initial_state, qubit_mapping=mapping, mp2_reduction=True)

    ######################## VQE RESULT ###############################
        # setup a classical optimizer for VQE
    max_eval = 200
    optimizer = COBYLA(maxiter=max_eval, disp=True, tol=1e-3)

    print('Doing VQE')
    algorithm = VQE(pauli_operator, var_op, optimizer,'paulis', initial_point=None, )
    result = algorithm.run(backend)
    var_energy = result['energy'] + coulomb_energy
    # vqe_params = result['opt_params']

    return var_energy


def run_exact(pauli_operator, coulomb_energy):

    exact_eigensolver = ExactEigensolver(pauli_operator, k=1)
    ret = exact_eigensolver.run()
    print('The electronic energy is: {:.12f}'.format(ret['eigvals'][0].real))
    print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real + coulomb_energy))
    ed_energy = ret['eigvals'][0].real + coulomb_energy

    return ed_energy

##Carefully upgrade terra to see if qasm/state-vector simulator perform quicker.##

#################### WALK ROOT DIR ############################
root_dir = 'IntegralData/vqe-data-master/Li2_cc-pVTZ/4_ORBITALS'

data_file_list = []
data_file_list_oe = []
for dirName, subdirList, fileList in os.walk(root_dir):
    print('Found directory: %s' % dirName)
    for fname in sorted(fileList):
        if fname.endswith('.yaml'):
            data_file_list.append(fname)
        if fname.endswith('.FOCK'):
            data_file_list_oe.append(fname)

#This 'del' is four the 4-orbital case since the OEs are missing for the distance 13...
#This should be resolved for the future
del data_file_list[-1]

#I added an x in front of the 10s for distance to force the sorting, else it sees 13 as 1
#If any distance is in the 100s (unlikely) then add a 'y' in front of dist in file name
###############################################################
############# Output Files to Plot stuff ###############
print(data_file_list)
Fout = open('Li2_ExactEnergies_wMP2_4-orbitals_061219.dat',"w")

###########################################################

#Variables that can be assigned outside of Loop
map_type = str('jordan_wigner')
truncation_threshold = 0.01

################# IBM BACKEND #####################
backend1 = Aer.get_backend('statevector_simulator')
backend2 = Aer.get_backend('qasm_simulator')
###################################################

@dask.delayed
def final(root_dir, file1, file2):
    spatial_orbitals, spin_orbitals, particle_num, qubits_num, dist, nuclear_energy, one_electron_import, two_electron_import = read_variables(
        root_dir, file1, file2)
    qop = get_molecular_qubit_operator(one_electron_import, two_electron_import, spatial_orbitals, truncation_threshold,
                                       map_type)
    # vqe_energy = run_vqe(qop, spin_orbitals, particle_num, qubits_num, nuclear_energy, map_type, backend1)
    vqe_energy = 0
    exact_energy = run_exact(qop, nuclear_energy)

    my_info = [dist, exact_energy, vqe_energy]

    return my_info




# print(output_data)
Fout.close()

if __name__ == "__main__":
    computations = []
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)

    for file1, file2 in zip(data_file_list, data_file_list_oe):
        computations.append(final(root_dir, file1, file2))
        # my_info_to_str = " ".join(str(e) for e in final_energies)
        # Fout.write(my_info_to_str + "\n")

    output_data = dask.compute(*computations, scheduler='distributed')

