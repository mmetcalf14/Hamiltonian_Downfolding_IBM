#Once debugged, put this in Jupyter notebook

from math import pi
import numpy as np
import scipy as sp

# importing Qiskit

from qiskit import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit.tools.visualization import plot_histogram, matplotlib_circuit_drawer
from scipy import linalg as las

def xx_unitary(qcirc,theta, qa, qi, qj):

    # This is the circuit to do simulate an X_iX_j exponentiated Pauli operator where i < j
    qcirc.h(qi)
    qcirc.h( qj)
    qcirc.cx(qj, qi)
    qcirc.crz(2.0 *theta,qa,qi)
    qcirc.cx(qj, qi)
    qcirc.h(qi)
    qcirc.h( qj)


def yy_unitary(qcirc,theta, qa, qi, qj):

    # This is the circuit to do simulate an Y_iY_j exponentiated Pauli operator where i < j
    qcirc.rx(pi/2.,qi)
    qcirc.rx(pi / 2., qj)
    qcirc.cx(qj, qi)
    qcirc.crz(2.0 * theta,qa,qi)
    qcirc.cx(qj, qi)
    qcirc.rx(-pi/2.,qi)
    qcirc.rx(-pi / 2., qj)


def zz_unitary(qcirc,theta, qa, qi, qj):

    # This is the circuit to do simulate an Z_iZ_j exponentiated Pauli operator where i < j
    #I had to add a multiple of 2 for the CZ and now the eigenvalues match the unitaries in Miro's code

    qcirc.cx(qj, qi)
    qcirc.crz(2.0 * theta,qa,qi)
    qcirc.cx(qj, qi)


def z_unitary(qcirc,theta, qa, qi):

    qcirc.crz(2.0 * theta, qa, qi)


def unitary_specifiedtrotter(qcirc, h0, h1, h2, h3, h4, qa, q0, q1):

    z_unitary(qcirc, h0, qa, q0)
    z_unitary(qcirc, h1, qa, q1)
    xx_unitary(qcirc, h2, qa, q0, q1)
    yy_unitary(qcirc, h3, qa, q0, q1)
    # Expectation value of the following term can be computed classically with HF state.
    zz_unitary(qcirc, h4, qa, q0, q1)


    # print(qcirc)
    return 0

# Now begins main()

backend = Aer.get_backend('qasm_simulator')
# backend = Aer.get_backend('unitary_simulator')

# qr = QuantumRegister(2,name='qr')
# a = QuantumRegister(1, name='a')
# cr = ClassicalRegister(1)
# Circuit = QuantumCircuit(qr,cr)
# Circuit.add_register(a)

# Hamiltonian Values - Interestingly the entangling XX and YY terms are dominant when R is large - why?
# R = 1.55
hI = -0.2265
h0 = 0.1843
h1 = -0.0549
h2 = 0.1165
h3 = 0.1165
h4 = 0.4386


#Strangly there is a t_0 being used for U(2^k t_0) = (e^i\theta Ht_0)^(2^k)
#remember is trotter expansion dt = t_0
t_0 = 9.830

# Digit of precision
bits = 8


# Circuit.h(a[0])
# Circuit.rz(0,a[0]) # This is identity, but will be needed in the circuit to get increased bits of precision
# Circuit.x(qr[0]) # Initialize HF state for two qubit system
# unitary_specifiedtrotter(Circuit,(2**bits)*h0*t_0, (2**bits)*h1*t_0, (2**bits)*h2*t_0, (2**bits)*h3*t_0,\
#                            (2**bits)*h4*t_0, a[0], qr[0], qr[1])
#
# # Circuit.u1(t_0 * hI * (2 ** bits), qr[0])
# Circuit.h(a[0])
# Circuit.measure(a[0], cr[0])
#
# # print('Circuit for least significant bit\n',Circuit)
# #Circuit.draw(output='mpl')
#
#
# alg = execute(Circuit, backend)
# result = alg.result()
# count = result.get_counts(Circuit)
# print(count)
# x = 1 if count['1'] > count['0'] else 0
# c1 = count['1']/1024
# c2 = count['0']/1024
# print('The distribution of measurements are {} with population percentage {}'.format(x, c1)) if count['1'] > count['0']\
#     else print('The distribution of measurements are {} with population percentage {}'.format(x, c1))

# Execute unitary_sim
# result = execute(Circuit, backend).result()
# unitary = result.get_unitary(Circuit)
# print("Circuit unitary:\n", unitary)
# eval, evec = las.eigh(unitary)
# print(eval[0])

binary_phase = []
x = 0
omega_coef = 0
for k in range(bits,0, -1):
    it_num = k-1
    print('exponent: ', it_num)
    omega_coef /= 2.0
    # omega_coef = 0
    # for j in range(1,k-1):
    #     print('j = ',j)
    #     omega_coef += -pi*binary_phase[j-1]/(2**j)
    print(' {} is phase kickback'.format(omega_coef))
    qr = QuantumRegister(2, name='qr')
    a = QuantumRegister(1, name='a')
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    qc.add_register(a)

    qc.h(a[0])
    qc.u1(-pi*2.*omega_coef,a[0])
    qc.x(qr[0]) # Initialize HF state for two qubit system
    unitary_specifiedtrotter(qc,(2**it_num)*h0*t_0, (2**it_num)*h1*t_0, (2**it_num)*h2*t_0, (2**it_num)*h3*t_0,\
                               (2**it_num)*h4*t_0, a[0], qr[0], qr[1])

    # qc.u1(t_0 * hI * (2 ** bits), a[0])
    qc.h(a[0])
    qc.measure(a[0], cr[0])

    alg = execute(qc, backend)
    result = alg.result()
    count = result.get_counts(qc)
    x = 1 if count['1'] > count['0'] else  0

    omega_coef = omega_coef + x / 2
    print('phase is {} with a bit {} for k = {}'.format(omega_coef, x, k))
    binary_phase.insert(0,x)

E = (omega_coef-3*pi)/t_0
print(E+hI)
print(binary_phase)














