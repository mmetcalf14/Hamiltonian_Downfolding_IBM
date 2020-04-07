from qiskit import Aer, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from math import pi

dt = 0.001
qr = QuantumRegister(4)
cr = ClassicalRegister(4)
Circuit = QuantumCircuit(qr,cr)
backend = Aer.get_backend('qasm_simulator')


# Circuit.ry(0.9161744058,qr[1])
# Circuit.u1(pi/2.,qr[1]
# Circuit.rz(1.,qr[0])
#Circuit.rx(dt,qr[0])
#Circuit.measure(qr[0], cr[0])

Circuit.ry(pi,qr[0])
# Circuit.h(qr[2])
# Circuit.h(qr[3])
Circuit.cx(qr[0],qr[2])
Circuit.cx(qr[0],qr[3])
# Circuit.ry(-pi,qr[0])
# Circuit.h(qr[2])
# Circuit.h(qr[3])
Circuit.measure(qr[0], cr[0])
Circuit.measure(qr[1], cr[1])
Circuit.measure(qr[2], cr[2])
Circuit.measure(qr[3], cr[3])


print(Circuit)
alg = execute(Circuit, backend)
result = alg.result()
count = result.get_counts(Circuit)
print(count)
# state_vec = result.get_statevector(Circuit)
# print(result)
# unitary = result.get_unitary(Circuit)
# print(unitary)



# Circuit.rx(pi/2.,qr[2])
# Circuit.rx(pi/2.,qr[3])
# Circuit.rz(0.0,qr[2])
# Circuit.rz(0.0,qr[3])
# Circuit.rz((-0.7)*dt, qr[0])
# Circuit.rz((-0.5)*dt, qr[1])
# Circuit.ry(pi,qr[0])
# Circuit.ry(pi,qr[1])
# Circuit.cx(qr[0],qr[1])
# Circuit.rz((0.5)*dt, qr[1])
# Circuit.cx(qr[0],qr[1])
# Circuit.ry(-pi,qr[0])
# Circuit.ry(-pi,qr[1])
# Circuit.rx(-pi,qr[0])
# Circuit.rx(-pi,qr[1])
# Circuit.cx(qr[0],qr[1])
# Circuit.rz((0.9)*dt, qr[1])
# Circuit.cx(qr[0],qr[1])
# Circuit.rx(pi,qr[0])
# Circuit.rx(pi,qr[1])
# Circuit.cx(qr[0],qr[1])
# Circuit.rz((0.6)*dt, qr[1])
# Circuit.cx(qr[0],qr[1])
# Circuit.measure(qr[2], cr[0])
# Circuit.measure(qr[3], cr[1])