from qiskit.chemistry import FermionicOperator
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import  NumPyEigensolver
from qiskit import Aer, BasicAer

import numpy as np
import yaml as yml
from yaml import SafeLoader as Loader
import Load_Hamiltonians as lh
import time


def bittest(k, l):
    """
        k (int) - fermion basis integer
        l (int) - orbital or site
        return (Bool) - occupied or unoccupied
    """
    return k & (1 << l)

#################
#backend1 = Aer.get_backend('statevector_sipmulator')
backend2 = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(shots=2**10, backend=backend2)
#################

#Importing data generated from NW Chem to run experiments
root_dir = '/Users/mmetcalf/Dropbox/Quantum Embedding/Codes/Lithium_Downfolding/Qiskit Chem/Hamiltonian_Downfolding_IBM/IntegralData/BeH2 RHF/'
NW_data_file = str(root_dir+'YAML/BeH2-E1.yaml')
doc_nw = open(NW_data_file,'r')
data = yml.load(doc_nw,Loader)

#Initialize Variables
map_type = str('jordan_wigner')
n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
print('repulsion: ',nuclear_repulsion_energy)
n_orbitals = 2*n_spatial_orbitals
n_particles = data['integral_sets'][0]['n_electrons']
n_alpha = int(n_particles/2)
n_beta = int(n_particles/2)
n_occ = n_particles/2
print('nocc ', n_occ)

hf_energy = data['integral_sets'][0]['scf_energy']['value']

###########################################################

one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

one_electron_spatial_integrals, two_electron_spatial_integrals = lh.get_spatial_integrals_noperm(one_electron_import,two_electron_import,n_spatial_orbitals)
h1, h2 = lh.convert_to_spin_index(one_electron_spatial_integrals,two_electron_spatial_integrals, n_spatial_orbitals)

print('Getting Pauli Operator')
fop = FermionicOperator(h1, h2)
qop_paulis = fop.mapping(map_type)

###########################################
#Exact Result to compare to: This can also be obtained via the yaml file once we verify correctness.
#Don't use trunctated Hamiltonian
print('Getting energy')
exact_eigensolver = NumPyEigensolver(qop_paulis, k=15)
ret = exact_eigensolver.run()
print('The electronic energy is: {:.12f}'.format(ret['eigvals'][0].real))
print('The total FCI energy is: {:.12f}'.format(ret['eigvals'][0].real + nuclear_repulsion_energy))
print('First excited state is: {:.12f}'.format(ret['eigvals'][1].real + nuclear_repulsion_energy))
print('Second excited state is: {:.12f}'.format(ret['eigvals'][2].real + nuclear_repulsion_energy))
print('Third excited state is: {:.12f}'.format(ret['eigvals'][3].real + nuclear_repulsion_energy))
print('Fourth excited state is: {:.12f}'.format(ret['eigvals'][4].real + nuclear_repulsion_energy))
print(ret['eigvals'].real + nuclear_repulsion_energy)
exact_wf = ret['eigvecs'][9].to_matrix()

count = 0
basis = {}
for key, val in enumerate(exact_wf):
    if np.abs(val) >= 0.01:
        print(key, bin(key), val)
        occ_up = []
        unocc_up = []
        occ_dn = []
        unocc_dn = []
        for i in range(2*n_spatial_orbitals):
            if bittest(key, i) and i >= n_occ and i < n_spatial_orbitals:
                occ_up.append(i)
            elif bittest(key, i) and i >= (n_spatial_orbitals+ n_occ) and i < (2*n_spatial_orbitals):
                occ_dn.append(i)
            elif not bittest(key, i) and i < n_occ:
                unocc_up.append(i)
            elif not bittest(key, i) and i >= n_spatial_orbitals and i < (n_spatial_orbitals+ n_occ):
                unocc_dn.append(i)

        basis[key] = {'occ up':occ_up, 'unocc up':unocc_up, 'occ dn':occ_dn, 'unocc dn':unocc_dn, 'val':val}
        count += 1
print(basis)
print('count: ', count)
###########################################