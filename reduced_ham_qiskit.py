def bitTest(m, n):
    eval = m & (1 << n)
    return eval


def getindices(num_qubits, num_particles):
    indices_lst = []
    minimum = 0
    maximum = 0
    for i in range(1, num_particles + 1):
        minimum += (2 ** (i - 1))
        maximum += (2 ** (num_qubits - i))

    for j in range(minimum, maximum + 1):
        nbit_up = 0
        nbit_down = 0
        for q in range(num_qubits):
            if bitTest(j, q) and ((q % 2) == 0):
                nbit_up += 1
            if bitTest(j, q) and ((q % 2) != 0):
                nbit_down += 1
        if nbit_up == (num_particles / 2) and nbit_down == (num_particles / 2):
            indices_lst.append(j)

    return indices_lst