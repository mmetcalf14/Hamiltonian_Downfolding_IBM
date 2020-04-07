from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD

class UCCSD2(UCCSD):
    def __init__(self, circ):
        self.circ = circ
