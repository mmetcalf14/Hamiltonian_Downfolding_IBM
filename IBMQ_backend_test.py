from qiskit import IBMQ, Aer, QuantumRegister, QuantumCircuit, ClassicalRegister
from math import pi
from qiskit.aqua import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, TensoredMeasFitter
import numpy as np
from qiskit.validation import BaseModel
from qiskit.providers.models import BackendProperties
from qiskit.providers.aer import noise
import yaml
IBMQ.enable_account(token='f3e59f3967991477fc6a8413858524079a26a6e512874242a4ef41532cceca99b646d05af0e29fe234bd4aa5f4d491b6a5cd90734b124f5e8877d53884eafb74')

provider = IBMQ.get_provider(hub='ibm-q-ornl')
print(provider.backends())
device = provider.get_backend('ibmq_boeblingen')
backend = Aer.get_backend('qasm_simulator')
backend2 = Aer.get_backend('statevector_simulator')

# We will add the coupling map manually into the noise model
coupling_map = device.configuration().coupling_map
print(coupling_map)
print(device)
device_properties = device.properties()
noise_model = noise.device.basic_device_noise_model(device_properties)
basis_gates = noise_model.basis_gates
print(basis_gates)
print(device.status())
permitted_gates= device.configuration().basis_gates
print(permitted_gates)
print('Instantiating Quantum Instannce')

# quantum_instance = QuantumInstance(backend=backend, coupling_map=coupling_map, basis_gates=permitted_gates,
#                                    optimization_level=2,noise_model=noise_model, measurement_error_mitigation_cls=CompleteMeasFitter,
#                                    measurement_error_mitigation_shots=2**9)
# quantum_instance = QuantumInstance(backend=backend, coupling_map=coupling_map, basis_gates=permitted_gates,
#                                    optimization_level=2,noise_model=None, measurement_error_mitigation_cls=CompleteMeasFitter,
#                                    measurement_error_mitigation_shots=2**9)
quantum_instance1 = QuantumInstance(shots=2**13, backend=backend, coupling_map=coupling_map, basis_gates=permitted_gates,
                                   optimization_level=2,noise_model=None, measurement_error_mitigation_cls=None,
                                   measurement_error_mitigation_shots=2**12)
quantum_instance2 = QuantumInstance(shots=2**13, backend=backend, coupling_map=coupling_map, basis_gates=permitted_gates,
                                   optimization_level=2,noise_model=noise_model, measurement_error_mitigation_cls=None,
                                   measurement_error_mitigation_shots=2**12)
# quantum_instance3 = QuantumInstance(shots=2**13, backend=backend, coupling_map=coupling_map, basis_gates=permitted_gates,
#                                    optimization_level=2,noise_model=noise_model, measurement_error_mitigation_cls=TensoredMeasFitter,
#                                    measurement_error_mitigation_shots=2**12)
# quantum_instance = QuantumInstance(backend=backend2, basis_gates=permitted_gates)
# dict = device_properties.to_dict()
# new_prop = BackendProperties.from_dict(dict_=dict)

#print(dict.qubits)


# with open('ibmq_boeblingen_device-properties_092019.yml', 'w') as outfile:
#     yaml.dump(device_properties.to_dict(), outfile, default_flow_style=False)

qr = QuantumRegister(2)
cr = ClassicalRegister(2)
circuit = QuantumCircuit(qr, cr)

circuit.h(qr[0])
circuit.cx(qr[0],qr[1])
circuit.u1(pi/4., qr[1])
circuit.cx(qr[0],qr[1])
circuit.h(qr[0])
circuit.measure(qr[0],cr[0])
circuit.measure(qr[1],cr[1])
print(circuit)

result1 = quantum_instance1.execute(circuits=circuit)
result2 = quantum_instance2.execute(circuits=circuit)
# result3 = quantum_instance3.execute(circuits=circuit)
counts = result1.get_counts()
noisy_counts = result2.get_counts()
print(counts)
print(noisy_counts)
result_vec=np.zeros(4)
result_vec[0] = counts['00']
result_vec[1] = counts['01']
print(result_vec)
# print(result2.get_counts())
# print(result3.get_counts())
# result = quantum_instance.execute(circuits=circuit)
# statevector = result.get_statevector()
# print(np.square(np.abs(statevector)))
