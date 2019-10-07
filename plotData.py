import matplotlib.pyplot as plt

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
