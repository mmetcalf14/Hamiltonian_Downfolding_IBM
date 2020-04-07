from math import pi
import numpy as np
import scipy as sp

# importing Qiskit

from qiskit import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute

backend = Aer.get_backend('statevector_simulator')

# qr = QuantumRegister(2)
# cr = ClassicalRegister(2)
# Circuit = QuantumCircuit(qr,cr)
#
# Circuit.x(0)
# Circuit.h(0)
# Circuit.h(1)
# Circuit.cx(0,1)
# Circuit.rz(pi/2.,1)
# Circuit.cx(0,1)
# Circuit.h(0)
# Circuit.h(1)
# Circuit.rx(pi/2.,0)
# Circuit.rx(pi/2.,1)
# Circuit.cx(0,1)
# Circuit.rz(pi/2.,1)
# Circuit.cx(0,1)
# Circuit.rx(pi/2.,0)
# Circuit.rx(pi/2.,1)
# Circuit.measure(qr[0],cr[0])
# Circuit.measure(qr[1],cr[1])
# print(Circuit)
# alg = execute(Circuit,backend)
# results = alg.result()
# print(results.get_statevector(Circuit))

iswap = [[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]
I = [[1,0],[0,1]]
def phase_gate(x):
    gate = [[1,0],[0,np.exp(1j*x)]]
    return gate
init_state = [0,1,0,0]
circ = np.dot(iswap,np.dot(np.kron(phase_gate(pi/4.),I),np.kron(I,phase_gate(pi/4.))))
result = np.dot(circ,init_state)
print(result)