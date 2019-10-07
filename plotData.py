import matplotlib.pyplot as plt


class Plotting:

    # def sort_x(self, l1, l2):
    #     """
    #     :param l1: (list) the list representing the x coordinates.
    #     :param l2: (list) the list representing the y coordinates.
    #     :return: a nested list where the first element is l1 sorted in ascending order, and the second element is l2 sorted
    #     based on l1.
    #     """
    #     sorted_l1, sorted_l2 = [], []
    #     if len(l1) is 1:
    #         if l1[0] is min(l1):
    #             sorted_l1[0] = l1[0]
    #             sorted_l2[0] = l2[0]
    #         return [sorted_l1, sorted_l2]
    #     self.sort_x(l1[1:], l2[1:])
    #
    # sort_x([1, 3, 2], [1, 2, 3], [1, 2, 3])

    X, Y, Z = [], [], []
    for line in open('H2_VQEEnergies_wMP2_WSingles_7-orbitals_091119_with_noise.dat', 'r'):
        values = [float(s) for s in line.split()]

        X.append(values[0])
        Y.append(values[1])
        Z.append((values[2]))

    index = 0
    while index < len(X) - 1:
        if X[index] > X[index + 1]:
            temp_val = X[index]
            X[index] = X[index + 1]
            X[index + 1] = temp_val
            temp_val2 = Y[index]
            Y[index] = Y[index + 1]
            Y[index + 1] = temp_val2
            temp_val3 = Z[index]
            Z[index] = Z[index + 1]
            Z[index + 1] = temp_val3
        index += 1

    plt.plot(X, Y, "B", marker='o', label='exact energy')
    plt.plot(X, Z, 'm', marker='o', label='vqe energy')
    plt.legend()
    plt.show()
