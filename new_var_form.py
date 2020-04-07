import logging
import numpy as np
from qiskit.quantum_info.operators import pauli as p
from qiskit.aqua.operators import evolution_circuit
from qiskit.aqua.components.variational_forms import VariationalForm

logger = logging.getLogger(__name__)


class NVARFORM(VariationalForm):
    CONFIGURATION = {
        'name': 'new_var_form',
        'description': 'new_var_form Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'new_var_form_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'num_orbitals': {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 1
                },
                'num_particles': {
                    'type': ['array', 'integer'],
                    'default': [1, 1],
                    'contains': {
                        'type': 'integer'
                    },
                    'minItems': 2,
                    'maxItems': 2
                },
                'qubit_mapping': {
                    'type': 'string',
                    'default': 'parity',
                    'enum': ['jordan_wigner', 'parity', 'bravyi_kitaev']
                },
                'two_qubit_reduction': {
                    'type': 'boolean',
                    'default': True
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'HartreeFock',
                }
            },
        ],
    }

    def __init__(self, num_qubits, depth, num_orbitals, num_particles,
                 initial_state=None, qubit_mapping='parity', two_qubit_reduction=True):
        self.validate(locals())
        super().__init__()
        self._num_qubits = num_orbitals if not two_qubit_reduction else num_orbitals - 2
        self._num_qubits = self._num_qubits
        if self._num_qubits != num_qubits:
            raise ValueError('Computed num qubits {} does not match actual {}'
                             .format(self._num_qubits, num_qubits))
        self._depth = depth
        self._num_parameters = 7
        self._initial_state = initial_state
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction

        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]


    def construct_unitary(self, parameters):
        unitary_matrix = np.zeros((16, 16))
        #excitations = [1, 3, 4, 6, 8, 9, 10, 13, 15]
        excitations = [3, 4, 6, 8, 10, 13, 15]

        # singles_excitations = [1, 4, 6, 9]
        for key, val in enumerate(excitations):
            unitary_matrix[val][0], unitary_matrix[0][val] = parameters[key], -parameters[key]

        unitary_matrix = -1j * unitary_matrix / 4

        np.savetxt('10_0_unitary1', unitary_matrix, delimiter=',')

        return unitary_matrix

    def pauli_decomp(self, parameters):
        unitary_matrix = self.construct_unitary(parameters)
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

        "Get non zero elements"
        nonzero_pauli = []
        for pauli in pauli_basis:
            norm = np.sqrt(np.sum(np.square(pauli.to_matrix())))
            if norm is complex:
                ham_pauli_prod = np.dot(-1j * unitary_matrix, pauli.to_matrix())
            else:
                ham_pauli_prod = np.dot(unitary_matrix, pauli.to_matrix())
            trace = np.trace(ham_pauli_prod)
            if trace != complex(0, 0):
                # if trace.imag > 1e-12:
                #     trace *= -1j
                nonzero_pauli.append([trace/4, pauli])
        return nonzero_pauli


    def construct_circuit(self, parameters, q=None):
        pauli_ops = self.pauli_decomp(parameters)
        ev_circuit = evolution_circuit(pauli_ops, evo_time=1, num_time_slices=1, controlled=False, power=1,
                                              use_basis_gates=False, shallow_slicing=False)
        return ev_circuit


    @property
    def preferred_init_points(self):
        """Getter of preferred initial points based on the given initial state."""
        if self._initial_state is None:
            return None
        else:
            bitstr = self._initial_state.bitstr
            if bitstr is not None:
                return np.zeros(self._num_parameters, dtype=np.float)
            else:
                return None

