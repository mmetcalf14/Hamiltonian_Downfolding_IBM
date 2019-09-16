from qiskit.chemistry import FermionicOperator, QMolecule
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry import QMolecule as qm

from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua import Operator
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit import Aer

import numpy as np
import yaml as yml
from yaml import SafeLoader as Loader
import os

import Load_Hamiltonians as lh

#################### WALK ROOT DIR ############################
root_dir = '/Users/mmetcalf/Dropbox/Quantum Embedding/Codes/Lithium_Downfolding/Qiskit Chem/Hamiltonian_Downfolding_IBM/IntegralData/Li2_cc-pVTZ/7_ORBITALS/'

data_file_list = []
data_file_list_oe = []
for dirName, subdirList, fileList in os.walk(root_dir):
    print('Found directory: %s' % dirName)
    for fname in sorted(fileList):
        if fname.endswith('.yaml'):
            data_file_list.append(fname)
        if fname.endswith('.FOCK'):
            data_file_list_oe.append(fname)

#I added an x in front of the 10s for distance to force the sorting, else it sees 13 as 1
#If any distance is in the 100s (unlikely) then add a 'y' in front of dist in file name

############# Output Files to Plot stuff ###############
print(data_file_list)
Fout = open('H2_VQEEnergies_wMP2_WSingles_7-orbitals_091119.dat',"w")
Fout_op = open('H2_OptimalParams_wMP2_7-orbitals_091119.dat',"w")
#Fout = open('Li2_ExactEnergiesG&FE_wMP2_4-orbitals_052919.dat',"w")
###########################################################

#Variables that can be assigned outside of Loop
map_type = str('jordan_wigner')
truncation_threshold = 0.01

################# IBM BACKEND #####################
backend1 = Aer.get_backend('statevector_simulator',)
backend2 = Aer.get_backend('qasm_simulator')

#Add noise model here.
###################################################

output_data = []
#Looping over all of the yaml files in a particular directory and saving the energies from VQE and Exact
ind = 0
for file1, file2 in zip(data_file_list, data_file_list_oe):

    NW_data_file = str(os.path.join(root_dir,file1))

    OE_data_file = str(os.path.join(root_dir+'Orbital_Energies/',file2))

    try:
        doc = open(NW_data_file, 'r')
        doc_oe = open(OE_data_file,'r')
        data = yml.load(doc, Loader)
        content_oe = doc_oe.readlines()
    finally:
        doc.close()
        doc_oe.close()

    #Import all the data from a yaml file
    print('Getting data')
    n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
    nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
    n_orbitals = 2 * n_spatial_orbitals
    n_particles = data['integral_sets'][0]['n_electrons']
    dist = 2 * data['integral_sets'][0]['geometry']['atoms'][1]['coords'][2]
    print('Bond distance is {}'.format(dist))
    if map_type == 'parity':
        # For two-qubit reduction
        n_qubits = n_orbitals - 2
    else:
        n_qubits = n_orbitals

    active_occ = n_particles // 2
    active_virt = (n_orbitals - n_particles) // 2
    active_occ_list = np.arange(active_occ)
    active_occ_list = active_occ_list.tolist()
    # print('Orbitals {} are occupied'.format(active_occ_list))
    active_virt_list = np.arange(active_virt)
    active_virt_list = active_virt_list.tolist()

    # Populating the QMolecule class with the data to make calculations easier
    qm.num_orbitals = n_spatial_orbitals
    qm.num_alpha = n_particles // 2
    qm.num_beta = n_particles // 2
    qm.core_orbitals = 0
    qm.nuclear_repulsion_energy = nuclear_repulsion_energy
    qm.hf_energy = data['integral_sets'][0]['scf_energy']['value']
    ###################Get Orbital Energies from FOCK file########################
    orbital_energies = []
    found = False
    count = 0
    for line in content_oe:
        if 'Eigenvalues:' in line.split()[0]:
            found = True
        if found and len(line.split()) > 1:
            orbital_energies.append(float(line.split()[1]))
            count += 1
            if count >= n_spatial_orbitals:
                break
    qm.orbital_energies = orbital_energies
    print('OEs:\n',orbital_energies)
    ##############################################################################

    #Importing the integrals
    one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
    two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

    #Getting spatial integrals and spin integrals to construct Hamiltonian
    one_electron_spatial_integrals, two_electron_spatial_integrals = lh.get_spatial_integrals(one_electron_import,
                                                                                              two_electron_import,
                                                                                              n_spatial_orbitals)
    one_electron_spatial_integrals, two_electron_spatial_integrals = lh.trunctate_spatial_integrals(
        one_electron_spatial_integrals, two_electron_spatial_integrals, .001)
    h1, h2 = lh.convert_to_spin_index(one_electron_spatial_integrals, two_electron_spatial_integrals,
                                                  n_spatial_orbitals, truncation_threshold)
    #For the MP2 Calculation
    qm.mo_eri_ints = two_electron_spatial_integrals
    #Constructing the fermion operator and qubit operator from integrals data
    fop = FermionicOperator(h1, h2)
    qop_paulis = fop.mapping(map_type)
    qop = Operator(paulis=qop_paulis.paulis)

    #Get Variational form and intial state
    init_state = HartreeFock(n_qubits, n_orbitals, n_particles, map_type, two_qubit_reduction=False)
    var_op = UCCSD(num_qubits=n_qubits, depth=1, num_orbitals=n_orbitals, num_particles=n_particles, active_occupied=active_occ_list,\
               active_unoccupied=active_virt_list,initial_state=init_state,
                qubit_mapping=map_type, mp2_reduction=True, singles_deletion=False)
    print('There are {} params'.format(var_op.num_parameters))
    ######################## VQE RESULT ###############################
        # setup a classical optimizer for VQE
    max_eval = 500
    optimizer = COBYLA(maxiter=max_eval, disp=True, tol=1e-4)

    #Choosing initial params based on previous iteration
    if ind == 0:
        # initial_params = var_op._mp2_coeff
        initial_params = None
        ind += 1
    elif len(vqe_params) == var_op.num_parameters:
        initial_params = vqe_params
    else:
        initial_params = None


    print('Doing VQE')
    algorithm = VQE(qop_paulis,var_op,optimizer,'paulis', initial_point=initial_params)
    #VQE_Circ = algorithm.construct_circuit(dumpy_params, backend1)
    #print('The VQE circuit:\n',VQE_Circ)

    result = algorithm.run(backend1)
    vqe_energy = result['energy'] + nuclear_repulsion_energy
    vqe_params = result['opt_params']
    ###################################################################

    ################### EXACT RESULT ##################################
    exact_eigensolver = ExactEigensolver(qop, k=1)
    ret = exact_eigensolver.run()
    print('The electronic energy is: {:.12f}'.format(ret['eigvals'][0].real))
    print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real + nuclear_repulsion_energy))
    exact_energy = ret['eigvals'][0].real + nuclear_repulsion_energy

    ###################################################################
    # The outputs to plot
    my_info = [dist,exact_energy,vqe_energy]
    param_info = [dist,vqe_params]
    output_data.append(my_info)
    my_info_to_str = "\t".join(str(e) for e in my_info)
    params_to_str = "\t".join(str(e) for e in param_info)
    Fout.write(my_info_to_str + "\n")
    Fout_op.write(params_to_str + "\n")

print(output_data)
Fout.close()
Fout_op.close()

# output_data[:,0] = np.sort(output_data[:,0],axis=0)
# print(output_data)