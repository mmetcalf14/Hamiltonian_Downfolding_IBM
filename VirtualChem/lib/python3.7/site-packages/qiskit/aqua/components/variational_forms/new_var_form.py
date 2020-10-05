import logging
import numpy as np
from qiskit import QuantumCircuit
from qiskit.aqua.aqua_globals import QiskitAquaGlobals
from qiskit.quantum_info.operators import pauli as p
from qiskit.aqua.operators import evolution_instruction
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit import QuantumRegister
from qiskit.tools import parallel_map
from itertools import product
import scipy.linalg as la


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
        validate_min('num_orbitals', num_orbitals, 1)
        if isinstance(num_particles, list) and len(num_particles) != 2:
            raise ValueError('Num particles value {}. Number of values allowed is 2'.format(
                num_particles))
        validate_in_set('qubit_mapping', qubit_mapping,
                        {'jordan_wigner', 'parity', 'bravyi_kitaev'})

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
        self._num_time_slices = 1

        self._support_parameterized_circuit = True

        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]
        self.unique_coeff = []
        self.groups_pauli = self.pauli_decomp(self.num_parameters)


    def unique_coeff_paulis(self, ceoff, coeff_and_pauli):
        unique_vals_dict = {}
        for i in ceoff:
            unique_vals_dict.update({i: []})

        for el in coeff_and_pauli:
            unique_vals_dict[el[0]].append(el[1])

        return unique_vals_dict


    def construct_unitary(self, parameters):
        unitary_matrix = np.zeros((16, 16))
        # excitations = [1, 3, 4, 6, 8, 9, 10, 13, 15]
        excitations = [3, 4, 6, 8, 10, 13, 15]
        # excitations=[6]
        # param = [3, 6, 9, 12, 18, 24, 33, 36, 48, 66, 72, 96, 129, 132, 144, 192]
        # singles_excitations = [1, 4, 6, 9]

        param = 1e-8 * np.random.rand(7)
        for key, val in enumerate(excitations):
            unitary_matrix[val][0], unitary_matrix[0][val] = param[key], -param[key]

        unitary_matrix = -1j * unitary_matrix / 4

        np.savetxt('10_0_unitary1', unitary_matrix, delimiter=',')

        return unitary_matrix

    def construct_single_param_unitary_lst(self, len_parameters):
        unitary_matrix = np.zeros((16, 16))
        unitary_matrix_test = np.zeros((16, 16))
        broken_down_unitary_lst = []

        excitations = [3, 4, 6, 8, 10, 13, 15]
        parameter = 1e-8 * np.random.rand(len_parameters)

        for i in range(len(parameter)):
            unitary_matrix[excitations[i]][0], unitary_matrix[0][excitations[i]] = parameter[i], -parameter[i]
            unitary_matrix = -1j * unitary_matrix / 4
            broken_down_unitary_lst.append(unitary_matrix)
            unitary_matrix = np.zeros((16, 16))

        for key, val in enumerate(excitations):
            unitary_matrix_test[val][0], unitary_matrix_test[0][val] = parameter[key], -parameter[key]
        unitary_matrix_test = -1j * unitary_matrix_test / 4

        eval_sum_matrix, evec_sum_matrix = la.eigh(sum(broken_down_unitary_lst))
        eval_test_matrix, evec_test_matrix = la.eigh(unitary_matrix_test)

        print("eigenvalues of decontructed matrix: ", eval_sum_matrix, "\neigenvalues of test matrix: ", eval_test_matrix)
        return broken_down_unitary_lst


    def pauli_decomp(self, len_parameters):

        deconstructed_unitary_lst = self.construct_single_param_unitary_lst(len_parameters)
        # unitary_matrix = self.construct_unitary(parameters)
        input_array_lst = []
        pauli_basis = []
        coeff = []

        # this is equivalent to doing num_qubits for loops, each of which iterates through the loop 2 times
        for i in product(range(2), repeat=self.num_qubits):
            input_array_lst.append(np.array(i))

        for i in range(len(input_array_lst)):
            for j in range(len(input_array_lst)):
                pauli = p.Pauli(input_array_lst[i], input_array_lst[j])
                norm = np.sqrt(np.sum(np.square(pauli.to_matrix())))
                pauli_basis.append(pauli)

        grouped_paulis = []

        "Get non zero elements"
        for unitary in deconstructed_unitary_lst:
            nonzero_pauli = []
            for pauli in pauli_basis:
                norm = np.sqrt(np.sum(np.square(pauli.to_matrix())))
                if norm is complex:
                    ham_pauli_prod = np.dot(-1j * unitary, pauli.to_matrix())
                else:
                    ham_pauli_prod = np.dot(unitary, pauli.to_matrix())
                trace = np.trace(ham_pauli_prod)
                if trace != complex(0, 0):
                    if np.abs(trace.imag) > 1e-12:
                        trace *= -1j
                    nonzero_pauli.append([trace / 4, pauli])
                    coeff.append(trace / 4)
            grouped_paulis.append(nonzero_pauli)
        print("grouped pualis", grouped_paulis)
                # print("nonzero pauli",nonzero_pauli)

        # nonzero_pauli = []
        # for pauli in pauli_basis:
        #     norm = np.sqrt(np.sum(np.square(pauli.to_matrix())))
        #     if norm is complex:
        #         ham_pauli_prod = np.dot(-1j * unitary_matrix, pauli.to_matrix())
        #     else:
        #         ham_pauli_prod = np.dot(unitary_matrix, pauli.to_matrix())
        #     trace = np.trace(ham_pauli_prod)
        #     if trace != complex(0, 0):
        #         # if trace.imag > 1e-12:
        #         #     trace *= -1j
        #         nonzero_pauli.append([trace / 4, pauli])
        #         coeff.append(trace / 4)
        #     # print("nonzero pauli",nonzero_pauli)
        # self.unique_coeff = np.unique(np.array(coeff)).tolist()

        return grouped_paulis


    def construct_circuit(self, parameters, q=None):
        ###doesn't work when I pass in parameters
        print("parameters ",parameters)
        pauli_ops_lst = self.groups_pauli
        print("pauliops ",pauli_ops_lst)
        print(self.unique_coeff)
        # print(self.unique_coeff_paulis(self.unique_coeff, pauli_ops))

        param_and_op_lst = []
        for i in range(len(parameters)):
            qubit_op = WeightedPauliOperator(pauli_ops_lst[i])
            qubit_op = qubit_op * -1j
            param_and_op_lst.append((qubit_op, parameters[i]))

        print("param and op list", param_and_op_lst)


        # qubit_op = WeightedPauliOperator(pauli_ops)
        # qubit_op = qubit_op * -1j

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
            print("circuit type: ", circuit)
        else:
            circuit = QuantumCircuit(q)
        # circuit = QuantumCircuit(q)

        # param_and_op_lst = []
        # for param in parameters:
        #     param_and_op_lst.append((qubit_op, param))

        result = parallel_map(NVARFORM._construct_circuit_for_each_param,
                              param_and_op_lst, task_args=(q, self._num_time_slices),
                              num_processes=aqua_globals.num_processes)

        for circ in result:
            circuit.data += circ


        ### attempting the old method
        # if q is None:
        #     q = QuantumRegister(self._num_qubits, name='q')
        # pauli_ops = self.pauli_decomp(parameters)
        # qop = WeightedPauliOperator(pauli_ops)
        # circuit = qop.evolve(state_in=None, evo_time=1,
        #                           num_time_slices=1)

        return circuit

    @staticmethod
    def _construct_circuit_for_each_param(op_and_param, qr, num_time_slices):
        qop, parameters = op_and_param
        quantum_circ = qop.evolve(state_in=None, evo_time=parameters,
                                  num_time_slices=num_time_slices, quantum_registers=qr)

        return quantum_circ

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


# Global instance to be used as the entry point for globals.
aqua_globals = QiskitAquaGlobals()  # pylint: disable=invalid-name
