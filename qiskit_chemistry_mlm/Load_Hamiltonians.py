import numpy as np
#from itertools import permutations
#import Permutations as pm

def get_spatial_integrals(one_electron,two_electron,n_o):
    one_electron_spatial_integrals = np.zeros((n_o, n_o))
    two_electron_spatial_integrals = np.zeros((n_o, n_o, n_o, n_o))
    # populating a one-e spatial hamiltonian
    for ind, val in enumerate(one_electron):
        # This is because python index starts at 0
        i = int(val[0] - 1)
        j = int(val[1] - 1)
        one_electron_spatial_integrals[i, j] = val[2]
        if i != j:
            one_electron_spatial_integrals[j, i] = val[2]

    # populating a two-electron spatial hamiltonian
    for ind, val in enumerate(two_electron):
        i = int(val[0]-1)
        j = int(val[1]-1)
        k = int(val[2]-1)
        l = int(val[3]-1)
        my_ind = [i,j,k,l]
        # perm = permutations(my_ind)
        # perm = pm.perm_unique(my_ind)
        # print(my_ind)
        two_electron_spatial_integrals[i, j, k, l] = val[4]
        if two_electron_spatial_integrals[k, l, i, j] == 0:
            two_electron_spatial_integrals[k, l, i, j] = val[4]
            #print('First If: ', [k, l, i, j])
        if two_electron_spatial_integrals[i, j, l, k] == 0:
            two_electron_spatial_integrals[i, j, l, k] = val[4]
            #print('Second If: ', [i, j, l, k])
        if two_electron_spatial_integrals[l, k, i, j] == 0:
            two_electron_spatial_integrals[l, k, i, j] = val[4]
           # print('Third If: ', [l, k, i, j])
        if two_electron_spatial_integrals[j, i, k, l] == 0:
            two_electron_spatial_integrals[j, i, k, l] = val[4]
            #print('Fourth If: ', [j, i, k, l])
        if two_electron_spatial_integrals[k, l, j, i] == 0:
            two_electron_spatial_integrals[k, l, j, i] = val[4]
           # print('Fifth If: ', [k, l, j, i])
        if two_electron_spatial_integrals[j, i, l, k] == 0:
            two_electron_spatial_integrals[j, i, l, k] = val[4]
            #print('Sixth If: ', [j, i, l, k])
        if two_electron_spatial_integrals[l, k, j, i] == 0:
            two_electron_spatial_integrals[l, k, j, i] = val[4]
            #print('Seventh If: ', [l, k, j, i])

    return one_electron_spatial_integrals, two_electron_spatial_integrals

def get_spatial_integrals_noperm(one_electron,two_electron,n_o):
    one_electron_spatial_integrals = np.zeros((n_o, n_o))
    two_electron_spatial_integrals = np.zeros((n_o, n_o, n_o, n_o))
    # populating a one-e spatial hamiltonian
    for ind, val in enumerate(one_electron):
        # This is because python index starts at 0
        i = int(val[0] - 1)
        j = int(val[1] - 1)
        one_electron_spatial_integrals[i, j] = val[2]
        if i != j:
            one_electron_spatial_integrals[j, i] = val[2]

    # populating a two-electron spatial hamiltonian
    for ind, val in enumerate(two_electron):
        i = int(val[0]-1)
        j = int(val[1]-1)
        k = int(val[2]-1)
        l = int(val[3]-1)
        my_ind = [i,j,k,l]
        # perm = permutations(my_ind)
        # perm = pm.perm_unique(my_ind)
        # print(my_ind)
        two_electron_spatial_integrals[i, j, k, l] = val[4]

    return one_electron_spatial_integrals, two_electron_spatial_integrals


def convert_to_spin_index(one_electron, two_electron, n_o):
    h1 = np.block([[one_electron, np.zeros((int(n_o), int(n_o)))],
                   [np.zeros((int(n_o), int(n_o))), one_electron]])
    h2 = np.zeros((2 * n_o, 2 * n_o, 2 * n_o, 2 * n_o))

    for i in range(len(two_electron)):
        for j in range(len(two_electron)):
            for k in range(len(two_electron)):
                for l in range(len(two_electron)):

                    h2[i,j, k + n_o, l + n_o] = two_electron[i, j, k, l]
                    h2[i + n_o, j + n_o,k, l] = two_electron[i, j, k, l]

                    if i!=k and j!=l:
                        h2[i,j,k,l] = two_electron[i,j,k,l]
                        #print(h2[i,j,k,l])
                        h2[i + n_o, j + n_o, k + n_o, l + n_o] = two_electron[i, j, k, l]
    return h1, 0.5*h2

