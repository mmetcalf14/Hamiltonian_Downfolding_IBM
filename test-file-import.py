import os

root_dir = 'IntegralData/vqe-data-master/Li2_cc-pVTZ/4_ORBITALS/'
data_file = str(root_dir + 'DOWNFOLDED_ORBITAL_ENERGIES/Li2-4_ORBITALS-ccpVTZ-2_673.FOCK')
doc = open(data_file,'r')
content = doc.readlines()
print(content)
n_orb = 4
count = 0

orbital_energies = []


found = False
for line in content:
    if 'Eigenvalues:' in line.split()[0]:
        found =True
    if found and len(line.split()) > 1:
        #print(line.split()[1])
        orbital_energies.append(float(line.split()[1]))
        count += 1
        print(' {} is the count'.format(count))
        if count >= n_orb:
            break



print(orbital_energies)
doc.close()