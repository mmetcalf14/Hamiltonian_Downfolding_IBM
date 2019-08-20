
from math import pi
import numpy as np
import scipy as sp

# importing Qiskit

from qiskit import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

# We first define controlled gates used in the IPEA
def cu1fixed(qProg, c, t, a):
    qProg.u1(-a, t)
    qProg.cx(c, t)
    qProg.u1(a, t)
    qProg.cx(c, t)

def cu5pi8(qProg, c, t):
    cu1fixed(qProg, c, t, -5.0*pi/8.0)

backend = Aer.get_backend('statevector_simulator')
# backend = Aer.get_backend('qasm_simulator')
# We then prepare quantum and classical registers and the circuit
qr = QuantumRegister(2)
cr = ClassicalRegister(1)
Circuit = QuantumCircuit(qr,cr)


# Apply IPEA
Circuit.h(qr[0])
cu5pi8(Circuit, qr[0], qr[1])
# for i in range(8):
#     cu5pi8(Circuit, qr[0], qr[1])
# Circuit.u1(-3*pi/4,qr[0])
Circuit.h(qr[0])

#Circuit.measure(qr[0], cr[0])
print('Circuit for least significant bit',Circuit)

alg = execute(Circuit, backend)
result = alg.result()
print(result)
vec = result.get_statevector(Circuit)
vecx = np.conj(vec)
print(vec)
#plot_histogram(result.get_counts())
'''
Circuit.h(qr[0])
Circuit.u1(-pi/4., qr[1])
Circuit.cx(qr[0], qr[1])
Circuit.u1(pi/4., qr[1])
Circuit.cx(qr[0], qr[1])
Circuit.h(qr[0])
Circuit.measure(qr[0], cr[0])
print(Circuit)
alg = execute(Circuit, backend)
result = alg.result()
count = result.get_counts(Circuit)
print(count)
# state_vec = result.get_statevector(Circuit)
# vecx = np.conj(state_vec)
# print(np.outer(vecx,state_vec))
'''