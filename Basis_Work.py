# created on Tuesday, March 31, 2020
#
# @author - Mekena Metcalf
# @function - Create basis indexing scheme to map fermions states to qubit states
#

import numpy as np
import scipy.linalg as la
from math import factorial, sqrt, log2, ceil
from itertools import product
import warnings

def comb(x, y):
    return factorial(x)/(factorial(y)*factorial(x-y))

def special_comb(P,H):
    return factorial(P) / (factorial(H) * factorial(P+H))

class Basis():

    def __init__(self, num_spatial_orbitals, num_spin_up, num_spin_down, basis_type='block'):
        """ Constructor.

        This class creates a basis representation to map fermions with
        a specified number of particles to qubits.
        See paper -- eventually

        Args:
            num_spatial_orbitals (int) - number of molecular orbitals or lattice sites in system
            num_spin_up (int) - number of spin up fermions
            num_spin_down (int) - number of spin down fermions

        """
        print(basis_type)
        if basis_type == 'block' or basis_type == 'interleaved':
            self._basis_type = basis_type
        else: raise ValueError('Not a valid basis representation')

        self._num_spatial_orbitals = num_spatial_orbitals
        self._num_spin_up = num_spin_up
        self._num_spin_down = num_spin_down
        self._max_count_up = comb(num_spatial_orbitals,num_spin_up)

        self._excitations_up = self.generate_excitation_list(num_spatial_orbitals,num_spin_up)
        self._excitations_down = self.generate_excitation_list(num_spatial_orbitals, num_spin_down)
        self._spin_up_basis = self.create_single_particle_basis(num_spatial_orbitals,num_spin_up, excitations=self._excitations_up)
        self._spin_down_basis = self.create_single_particle_basis(num_spatial_orbitals,num_spin_down, excitations=self._excitations_down)

        # spin up dictionary -> function
        # spin down dictionary -> function
        # total spin basis (block or interleaved) -> function

    @staticmethod
    def bittest(k,l):
        """
            k (int) - fermion basis integer
            l (int) - orbital or site
            return (Bool) - occupied or unoccupied
        """
        return k & (1<<l)

    @staticmethod
    def bitclr(k,l):
        """
            k (int) - fermion basis integer
            l (int) - orbital or site
            return (Bool) - integer with lth bit set to zero
        """
        return k & ~(1 << l)

    @staticmethod
    def bitset(k,l):
        """
            k (int) - fermion basis integer
            l (int) - orbital or site
            return (Bool) - integer with lth bit set to one
        """
        return k | (1 << l)

    @staticmethod
    def unique_integer(x,y):
        z = 0
        if x >= max(x,y):
            z = x**2 + x +y
        else:
            z = y**2+x
        return z

    @staticmethod
    def unique_integer_test(x,y):
        z = 0
        if x >= max(x,y):
            z = x + y
        else:
            z = 2*y+x
        return z

    def generate_excitation_list(self, M, N):
        """:M int - number of orbitals
            N (int) - number of particles
            return (dict) - all single particle excitations
        """
        max_iter = 0
        active_occ = N
        active_unocc = M - N
        if active_occ >= active_unocc:
            max_iter = active_unocc
        else:
            max_iter = active_occ

        excitations = {}
        for it in range(1, (max_iter + 1)):
            list = []
            for vec1 in product(range(active_occ), repeat=it):
                temp_vec1 = np.ndarray.tolist(np.array(vec1))
                if (len(set(vec1)) > it - 1) and temp_vec1 == sorted(temp_vec1):
                    for vec2 in product(range(active_occ, M), repeat=it):
                        temp_vec2 = np.ndarray.tolist(np.array(vec2))
                        if len(set(vec2)) > it - 1 and temp_vec2 == sorted(temp_vec2):
                            list.append(np.append(vec1, vec2))
            excitations[it] = list

        return excitations

    def create_single_particle_basis(self, M, N, excitations):
        """

        :param M: number of spatial orbitals
        :param N: number of fermions for single spin
        :param excitations: single particle excitations (all/FCI)
        :return: dictionary {fermion basis:index}
        """

        count = np.arange(0,N)
        minrange = sum([2**(i) for i in count])
        maxrange = sum([2**(M-i-1) for i in count])
        print(minrange, maxrange)
        max_count = comb(M,N)
        print('{} is the maximum value for the integer'.format(max_count))
        # FS stands for Fermi-Sea
        FS = N-1
        # Initialize dictionary with min range values
        basis_data = {'excitation': {str([0,0]):{'excitation num':0,'vec': [0,0],'basis': minrange, 'P':0, 'H':0, 'I':0, 'phase':1}}}
        #excitations = self.generate_excitation_list(M,N)
        iterate = 0
        for items in excitations.items():
            excitation_number = items[0]
            #print('excitation_number', excitation_number)
            ph_pairs = items[1]

            for val in ph_pairs:
                #print(ph_pairs)
                iterate += 1
                #print('iteration: ', iterate)
                basis = minrange
                H = 0
                P = 0
                I = 1
                phase = 1
                temp_unocc = 0
                for it in range(excitation_number):
                    occ = val[it]
                    unocc = val[it+excitation_number]
                    # print(occ,unocc)
                    # print('Should be HF',basis)
                    # operate on basis with bit ops
                    # store basis and parity
                    # I think the parity can be done from HF state without checking each bit
                    # some sort of mod i % F == 0 or 1
                    # get P and H
                    # get I
                    basis = self.bitclr(basis,occ)
                    basis = self.bitset(basis,unocc)

                    if self._basis_type == 'interleaved':
                        temp_basis = 0
                        for bt in range(M):
                            if self.bittest(basis,bt):
                                temp_basis += 2**(bt + factorial(bt))
                        basis = temp_basis
                    # P and H determined by summing binomial coeff
                    # This could be done other ways
                    if unocc > FS:
                        # This scales as the number of unoccupied orbitals choose the number of excited particles
                        P += (unocc-FS)
                        #P = self.unique_integer(unocc,temp_unocc)
                        #P += comb(M,unocc-N)
                    else: None

                    if occ <= FS and (FS-occ)%2==0:
                        H += (occ+1)
                        #H += comb(M,occ)
                        phase *= 1
                    elif occ <= FS and (FS-occ)%2!=0:
                        H += (occ +1)
                        #H += comb(M, occ)
                        phase *= -1
                    else: None

                    temp_unocc = unocc
                    #I = comb(M,P)
                # Solving this problem inherently requires factoring to determine the dimensions of the box
                Index = 4*(H-1)+ P
                #Index = max(P,H)**2 + max(P,H) +H - P +1
                #Index = ceil(log2(basis))
                # This is the correct index if one had quantum memory or an ability to reference
                # the index within a fully quantum algorithm
                # however this is not the case.
                correct_index = iterate

                # This index is generated using Szudzik pairing function
                # We use this function to get the total index that maps to a qubit state
                # This function produces integers that are too large
                #Index = self.unique_integer(P,H)
                #Index = self.unique_integer(H,P)
                basis_data['excitation'][str(val)]  = {'excitation num':excitation_number,'vec': val,'basis': basis, 'P': P, 'H': H , 'I': correct_index, 'phase':phase}
                print('excitation: ',val,' P: ', P, ' H: ',H, ' Pairing index: ',Index, ' Correct Index: ',correct_index)
        # minrange is the hartree-fock ground state
        # I want to construct the basis using single and double excitations out of this state
        # This requires having a index of the excitations which we currently do not have
        print(basis_data)
        return basis_data

    def create_total_basis(self):
        """
            return (dict) - total basis reference ind
        """
        total_dict = {}
        total_basis = 0
        up_vec = 0
        down_vec = 0
        for item_down, val_down in self._spin_down_basis['excitation'].items():
            for item_up, val_up in self._spin_up_basis['excitation'].items():
                if self._basis_type == 'block':
                    total_basis = val_up['basis'] + (2**self._num_spatial_orbitals)*val_down['basis']
                    up_vec = val_up['vec']
                    down_vec = [i+self._num_spatial_orbitals for i in val_down['vec']]

                elif self._basis_type == 'interleaved':
                    total_basis = val_up['basis'] + 2*val_down['basis']
                    up_vec = [2*i for i in val_up['vec']]
                    down_vec = [2*i +1 for i in val_down['vec']]

                phase = val_up['phase']*val_down['phase']
                index_up = val_up['I']
                index_down = val_down['I']

                total_index = (index_down)*self._max_count_up + index_up
                #total_index = self.unique_integer(index_down,index_up)
                print(index_down, index_up, total_index)
                #total_index = self._max_count_up*(index_down-1)+index_up
                #total_dict[total_basis] = {'excitation number': np.array([item_up, item_down]), 'excitation': {'up': val_up['vec'], 'down': val_down['vec']}, 'phase':phase, 'Itot':total_index}
                total_dict[total_basis] = {'excitation number': np.array([val_up['excitation num'], val_down['excitation num']]),
                                           'excitation': {'up': up_vec, 'down': down_vec}, 'phase': phase,
                                           'Itot': total_index}

        return total_dict

    def compute_varform_excitations(self, total_basis):
        single_excitations = []
        double_excitations = []

        for bas, info in total_basis.items():
            #print(info)
            #print(info['excitation number'][0], info['excitation number'][1])
            if info['excitation number'][0] == 1 and  info['excitation number'][1] == 0:
                excitation = info['excitation']['up']
                single_excitations.append([excitation, info['Itot']])

            elif info['excitation number'][0] == 0 and  info['excitation number'][1] == 1:
                excitation = info['excitation']['down']
                single_excitations.append([excitation, info['Itot']])

            elif info['excitation number'][0] == 1 and  info['excitation number'][1] == 1:
                excitation = [info['excitation']['up'][0], info['excitation']['down'][0],info['excitation']['up'][1], info['excitation']['down'][1]]
                double_excitations.append([excitation, info['Itot']])

            elif info['excitation number'][0] == 2:
                double_excitations.append([info['excitation']['up'],info['Itot']])

            elif info['excitation number'][1] == 2:
                double_excitations.append([info['excitation']['down'],info['Itot']])

            else: None

        print(single_excitations)
        print()
        print(double_excitations)
        return single_excitations, double_excitations