import numpy as np
from qiskit.quantum_info.operators import pauli as p

"Unitary Matrix"
unitary_matrix = np.zeros((16, 16))
vector = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

unitary_matrix[0][1], unitary_matrix[1][0] = 1, -1
unitary_matrix[0][2], unitary_matrix[2][0] = 1, -1
unitary_matrix[0][3], unitary_matrix[3][0] = 1, -1
unitary_matrix[0][4], unitary_matrix[4][0] = 1, -1
unitary_matrix[0][6], unitary_matrix[6][0] = 1, -1
unitary_matrix[0][8], unitary_matrix[8][0] = 1, -1
unitary_matrix[0][10], unitary_matrix[10][0] = 1, -1
unitary_matrix[0][13], unitary_matrix[13][0] = 1, -1
unitary_matrix[0][15], unitary_matrix[15][0] = 1, -1

unitary_matrix = -1j * unitary_matrix / 4


qubits = list(range(0, 4))
input_array_lst = []
pauli_basis = []

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                input_array_lst.append(np.array([i, j, k, l]))

for i in range(len(input_array_lst)):
    for j in range(len(input_array_lst)):
        pauli = p.Pauli(input_array_lst[i], input_array_lst[j])
        norm = np.sqrt(np.sum(np.square(pauli.to_matrix())))
        pauli_basis.append(pauli)

"Get coefficients"
coefficients = {}
non_zero_coeff = {}
nonzero_pauli_lst_matrix = []
nonzero_pauli_lst = []
nonzero_pauli = []
count = 0
for pauli in pauli_basis:
    ham_pauli_prod = np.dot(unitary_matrix, pauli.to_matrix())

    trace = np.trace(ham_pauli_prod)
    if trace != complex(0, 0):
        nonzero_pauli_lst_matrix.append(pauli.to_matrix())
        nonzero_pauli_lst.append(pauli.to_label())
        if trace.imag > 1e-12:
            trace *= -1j
        nonzero_pauli.append([trace/4, pauli])
    coefficients.update({pauli.to_label(): trace/4})
    if trace != complex(0, 0):
        non_zero_coeff.update({pauli.to_label(): trace/4})

