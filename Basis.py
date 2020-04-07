# created on Tuesday, March 31, 2020
#
# @author - Mekena Metcalf
# @function - Create basis indexing scheme to map fermions states to qubit states
#

import numpy as np
import scipy.linalg as la
from math import factorial
from itertools import product
import warnings

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

        print('inputs: ',M,N)
        count = np.arange(0,N)
        minrange = sum([2**(i) for i in count])
        maxrange = sum([2**(M-i-1) for i in count])
        print(minrange, maxrange)
        # FS stands for Fermi-Sea
        FS = N-1
        # Initialize dictionary with min range values
        basis_data = {'excitation': {'[0,0]' : {'vec': [0,0],'basis': minrange, 'P':0, 'H':0, 'I':0, 'phase':1}}}
        #excitations = self.generate_excitation_list(M,N)

        for items in excitations.items():
            excitation_number = items[0]
            #print('excitation_number', excitation_number)
            ph_pairs = items[1]
            # we need the excitation number to determine the number of ph ops
            # that need to be summed
            # For UCCSD we don't need the basis state
            # we only need the excitation to construct the unitary
            # and the integer corresponding to it.
            # hmm we do need the actual basis vector to project out the Ham
            # This can be easily done with other binary operations
            iterate = 0
            for val in ph_pairs:
                #print(ph_pairs)
                iterate +=1
                if iterate%10 == 0:
                    print(iterate)
                basis = minrange
                H = 0
                P = 0
                phase = 1
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

                    if unocc > FS:
                        P += unocc-FS
                    else: None

                    if occ <= FS and (FS-occ)%2==0:
                        H += occ
                        phase *= 1
                    elif occ <= FS and (FS-occ)%2!=0:
                        H += occ
                        phase *= -1
                    else: None

                Index = P + H
                basis_data['excitation'][str(val)]  = {'vec': val,'basis': basis, 'P': P, 'H': H , 'I': Index, 'phase':phase}
        # minrange is the hartree-fock ground state
        # I want to construct the basis using single and double excitations out of this state
        # This requires having a index of the excitations which we currently do not have
        return basis_data

    def create_total_basis(self):
        """
            return (dict) - total basis reference ind
        """
        total_dict = {}
        for item_down, val_down in self._spin_down_basis['excitation'].items():
            for item_up, val_up in self._spin_up_basis['excitation'].items():
                if self._basis_type == 'block':
                    total_basis = val_up['basis'] + (2**self._num_spatial_orbitals)*val_down['basis']
                elif self._basis_type == 'interleaved':
                    total_basis = val_up['basis'] + 2*val_down['basis']
                phase = val_up['phase']*val_down['phase']
                index_up = val_up['I']
                index_down = val_down['I']
                total_index = self.unique_integer(index_down,index_up)
                total_dict[total_basis] = {'excitation': np.array([val_up['vec'],val_down['vec']]), 'phase':phase, 'Itot':total_index}

        return total_dict