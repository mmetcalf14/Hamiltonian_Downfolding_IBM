import yaml as yml
from yaml import SafeLoader as Loader
import sys

root_dir = '/Users/mmetcalf/Dropbox/Quantum Embedding/Codes/Lithium_Downfolding/Qiskit Chem/Hamiltonian_Downfolding_IBM/IntegralData/Li2_cc-pVTZ/4_ORBITALS/'
NW_data_file = str(root_dir+sys.argv[1])
OE_data_file = str(root_dir+ sys.argv[2])
print(sys.argv[1],sys.argv[2])
doc_nw = open(NW_data_file,'r')
doc_oe = open(OE_data_file,'r')
data = yml.load(doc_nw,Loader)
content_oe = doc_oe.readlines()