import scipy as sp
import numpy as np


def canonical_eigh(M, S):
    """Return the eigenvectors and eigenvalues in the original basis with potentially singular S"""
    # Solve for spectrum with canonical orthogonalization
    overlapCutoff = 1.0e-8
    [evals, evecs] = sp.linalg.eigh(S)

    # Determine rank
    RankOverlap = np.sum(evals > overlapCutoff)
    nullity = len(evals) - RankOverlap
    # Cutoff
    evals = evals[nullity:]
    evecs = evecs[:, nullity:]

    # Make transformation matrix
    U = np.dot(evecs, np.diag(evals ** (-0.5)))

    # Transform the Hamiltonian and find the new eigenvectors
    M_prime = np.dot(np.conj(U).T, np.dot(M, U))
    S_prime = np.dot(np.conj(U).T, np.dot(S, U))

    e_vals, e_vecs = sp.linalg.eigh(M_prime, S_prime)
    # Transform eigenvectors back to original basis
    e_vecs = np.dot(U, e_vecs)

    return e_vals, e_vecs, U


def FindSymmTransform(M, S, val):
    """Find a transform to project onto the eigenspace corresponding to eigenvalue val of operator
    representation M with overlap S"""
    e_vals, e_vecs, U = canonical_eigh(M, S)
    indices = np.where(np.abs(e_vals - val) < 0.1)[0]
    U = e_vecs[:, indices]
    return U
