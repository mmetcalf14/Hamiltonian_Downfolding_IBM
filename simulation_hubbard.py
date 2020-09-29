#!/usr/bin/env python3

import math,qiskit

if __name__ == '__main__':

	jdt = 0.1
	qubits = list(range(4))
	shots = 8192
	steps = 128

	backend = qiskit.BasicAer.get_backend('qasm_simulator')

	U = 3;	
	t = -1;

#Corrected gates!!
	def circuit(qubits,jdt,steps):
		l = len(qubits)
		c = qiskit.QuantumCircuit(l,1)

#		#initial state = half filled	
#		c.x(qubits[0])
#		c.x(qubits[2])

#initial state
		c.x(qubits[0])
		c.x(qubits[1])
		c.x(qubits[2])

		#initial state = equal occupation
#		c.h(qubits[0])
#		c.h(qubits[1])
#		c.h(qubits[2])
#		c.h(qubits[3])
	
		for i in range(steps):
			c.ry(math.pi/2,0)
			c.ry(math.pi/2,1)
			c.cx(0,1)
			c.rz(jdt,1)
			c.cx(0,1)
			c.ry(-math.pi/2,0)
			c.ry(-math.pi/2,1)
		
			c.rx(-math.pi/2,0)
			c.rx(-math.pi/2,1)
			c.cx(0,1)
			c.rz(jdt,1)
			c.cx(0,1)
			c.rx(math.pi/2,0)
			c.rx(math.pi/2,1)

			c.ry(math.pi/2,2)
			c.ry(math.pi/2,3)
			c.cx(2,3)
			c.rz(jdt,3)
			c.cx(2,3)
			c.ry(-math.pi/2,2)
			c.ry(-math.pi/2,3)
		
			c.rx(-math.pi/2,2)
			c.rx(-math.pi/2,3)
			c.cx(2,3)
			c.rz(jdt,3)
			c.cx(2,3)
			c.rx(math.pi/2,2)
			c.rx(math.pi/2,3)

			for i in range(0, l):
				c.rz(U*2*jdt*.25, i)
#			Z0-Z3 INTERACTIONS
			c.cx(0, 3)
			c.rz(-U*2*jdt*.25, 3)
			c.cx(0, 3)
			c.cx(1, 2)
			c.rz(-U*2*jdt*.25, 2)
			c.cx(1, 2)

		c.measure(qubits[-1],0)
	#	c.measure_all()

		c.draw()
	
		return c

	circuits = []
	for i in range(steps):
		circuits.append(circuit(qubits,jdt,i))


	job = qiskit.execute(circuits,backend,shots=shots)
#	job = qiskit.execute(circuits,backend)

	result = job.result()

#	print(result.get_unitary(circuits, decimals=3))


	for i in range(steps):
#		counts = result.get_counts(i)()
#		print(counts)
		c = result.get_counts(i).get('0',0)
		p = (c+0.5)/(shots+1)
		u = math.sqrt(p*(1-p)/(shots+1))
		print(round(i*jdt,1),2*p-1,2*u)
