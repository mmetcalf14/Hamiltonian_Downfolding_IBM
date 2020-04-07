import numpy as np
from scipy import linalg as la
from math import pi
b = 8 # bits

# Hamilatonian coefficients from Table I. in Ref. [1] for R = 1.55 Å

g0 = -0.2265
g1 = 0.1843
g2 = -0.0549
g3 = 0.4386
g4 = 0.1165
g5 = 0.1165

t0 = 9.830

e = np.eye(2,2)
x = [[0 ,1],[ 1, 0]]
y = [[0, -1j],[ 1j, 0]]
z = [[1, 0],[ 0, -1]]

ee = np.kron(e, e)
ez = np.kron(z, e)

ze = np.kron(e, z)
xx = np.kron(x, x)
yy = np.kron(y, y)
zz = np.kron(z, z)

h = g0*ee + g1*ez + g2*ze + g3*zz + g4*yy + g5*xx
hez = g1*ez
hze = g2*ze
hzz = g3 *zz
hyy = g4 * yy
hxx = g5 * xx

evl, evc = la.eigh(h)
print(evc)
pz = [[1, 0],[ 0 ,0]] # projector to zero
po = [[0 ,0],[ 0, 1]] # projector to one

#pmat = [[1 ,-1],[ 1, 1]]
ryp = 1/np.sqrt(2)*np.array([[1 ,-1],[ 1, 1]]) # rotation about z by +pi/2
ryn = 1/np.sqrt(2)*np.array([[1, 1],[ -1, 1]]) # rotation about z by -pi/2

x0 = np.kron(np.kron(e, x), e)
rypa = np.kron(ryp, ee)
ryna = np.kron(ryn, ee)
poa = np.kron(po, ee)

v = [1, 0, 0, 0, 0, 0, 0, 0] # initial state


#u = la.expm(-1j * h * t0 * (2**b))
u = la.expm(-1j * h * t0 )
# u = np.dot(la.expm(-1j * hez * t0 * (2**b)), np.dot(la.expm(-1j * hxx * t0 * (2**b)),np.dot(la.expm(-1j * hze * t0 * (2**b)),\
#     np.dot(la.expm(-1j * hyy * t0 * (2**b)),la.expm(-1j * hzz * t0 * (2**b))))))
#u = np.dot(la.expm(-1j * hez * t0 ), np.dot(la.expm(-1j * hxx * t0 ),np.dot(la.expm(-1j * hze * t0 ),\
#    np.dot(la.expm(-1j * hyy * t0 ),la.expm(-1j * hzz * t0 )))))
# cu = np.kron(pz, ee) + np.kron(po, u)
# print(g3 * t0 * (2**b))
# print(u)
evals, evec = la.eigh(u)
print('Eigenvalue of matrix:',evec)
# omega_coef = 0
# phase_bits = []
# for k in range(b,0,-1):
#     print(k)
#     # phase kickback
#     omega_coef /= 2.
#     # a = -pi * sum(r. / 2. ^ (1:j-1));
#     rz = [[np.exp(-1j * 2* pi* omega_coef), 0],[0, np.exp(1j *2* pi* omega_coef)]]
#     rza = np.kron(rz, ee)
#
#     u = la.expm(-1j * h * t0 * (2 ** (k-1)))
#     # u = np.dot(la.expm(-1j * hez * t0 * (2 ** b)),np.dot(la.expm(-1j * hxx * t0 * (2 ** b)), np.dot(la.expm(-1j * hze * t0 * (2 ** b)), \
#     #     np.dot(la.expm(-1j * hyy * t0 * (2 ** b)),la.expm(-1j * hzz * t0 * (2 ** b))))))
#     cu = np.kron(pz, ee) + np.kron(po, u)
#     w = np.dot(ryna, np.dot(cu , np.dot( rza , np.dot(rypa ,np.dot( x0 , v)))))
#
#     p = np.dot(np.conj(w.transpose()),np.dot(poa,w)) # probability of measuring one
#     print(p)
#     x = round(p)
#     print('{} bit for k = {}'.format(x.real,k))
#
#     omega_coef = omega_coef + x.real / 2.


#phase_bits = []
#for k in range(0,b):
#    print(b-k)
#    # phase kickback
#    # a = -pi * sum(r. / 2. ^ (1:j-1));
#    omega_coef = 0
#    # for j in range(0,k-1):
#    #     print('j = ',j, k, phase_bits[j-1])
#    #     omega_coef += -pi*phase_bits[j-1]/(2**(j+1))
#    print(' {} is phase kickback'.format(omega_coef))
#    rz = [[np.exp(-1j *  omega_coef/2.), 0],[0, np.exp(1j * omega_coef/2.)]]
#    rza = np.kron(rz, ee)
#
#    # u = la.expm(-1j * h * t0 * (2 ** (b-k+1)))
#    u = np.dot(la.expm(-1j * hez * t0 * (2 ** (b-k))),np.dot(la.expm(-1j * hxx * t0 * (2 ** (b-k))), np.dot(la.expm(-1j * hze * t0 * (2 ** (b-k))), la.expm(-1j * hyy * t0 * (2 ** (b-k))))))
#    cu = np.kron(pz, ee) + np.kron(po, u)
#    w = np.dot(ryna, np.dot(cu , np.dot( rza , np.dot(rypa , np.dot( x0 , v)))))
#
#    p = np.dot(np.conj(w.transpose()),np.dot(poa,w)) # probability of measuring one
#
#    x = round(p).real
#    phase_bits.insert(0,x)
#    print('{} bit for k = {}'.format(x,k))
#
#
#s = -3*pi
#total_phase = 0
#for j in range(1, b):
#    total_phase += -pi * phase_bits[j - 1] / (2 ** j)
#
#E = (total_phase + s)/t0
#print('{} is the energy with bit string {}'.format(E, phase_bits))
