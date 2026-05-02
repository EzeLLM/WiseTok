"""
Numerical Linear Algebra: Solvers and Decompositions

Examples of LU decomposition, eigenvalue iteration, SVD, QR decomposition,
and solving linear systems Ax = b using NumPy.
"""

import numpy as np
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")


def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LU decomposition using Gaussian elimination with partial pivoting.
    A = PLU where P is permutation, L is lower triangular, U is upper triangular.

    Returns (P, L, U) such that PA = LU.
    """
    n = A.shape[0]
    A_copy = A.astype(float)
    P = np.eye(n)
    L = np.eye(n)
    U = np.zeros((n, n))

    for k in range(n):
        # Partial pivoting: find row with max value in column k
        max_idx = k + np.argmax(np.abs(A_copy[k:, k]))
        if max_idx != k:
            A_copy[[k, max_idx], :] = A_copy[[max_idx, k], :]
            P[[k, max_idx], :] = P[[max_idx, k], :]
            if k > 0:
                L[[k, max_idx], :k] = L[[max_idx, k], :k]

        if np.abs(A_copy[k, k]) < 1e-10:
            continue

        # Elimination
        for i in range(k + 1, n):
            L[i, k] = A_copy[i, k] / A_copy[k, k]
            A_copy[i, k:] -= L[i, k] * A_copy[k, k:]

    U = np.triu(A_copy)
    return P, L, U


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using LU decomposition.

    Algorithm:
    1. Decompose PA = LU
    2. Solve Ly = Pb (forward substitution)
    3. Solve Ux = y (back substitution)
    """
    P, L, U = lu_decomposition(A)
    Pb = P @ b

    # Forward substitution: Ly = Pb
    y = np.zeros_like(b, dtype=float)
    for i in range(len(b)):
        y[i] = (Pb[i] - L[i, :i] @ y[:i]) / L[i, i]

    # Back substitution: Ux = y
    x = np.zeros_like(b, dtype=float)
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x


def power_iteration(A: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> Tuple[float, np.ndarray]:
    """
    Power iteration to find largest eigenvalue and corresponding eigenvector.

    λ_max ≈ ||Av_k|| / ||v_k|| as k → ∞
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    lambda_prev = 0.0
    for iteration in range(max_iter):
        v_new = A @ v
        lambda_new = np.dot(v, v_new)
        v_new = v_new / np.linalg.norm(v_new)

        if np.abs(lambda_new - lambda_prev) < tol:
            break

        v = v_new
        lambda_prev = lambda_new

    return lambda_new, v


def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    QR decomposition using Gram-Schmidt orthogonalization.
    A = QR where Q is orthogonal (Q^T Q = I), R is upper triangular.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()
        # Orthogonalize against previous columns
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def svd_truncated(A: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncated SVD: A ≈ U[:, :rank] @ Σ[:rank, :rank] @ V[:rank, :].T
    Useful for low-rank approximation and denoising.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    idx = np.argsort(-s)[:rank]
    return U[:, idx], s[idx], Vt[idx, :]


def condition_number_analysis(A: np.ndarray) -> dict:
    """
    Analyze matrix conditioning: cond(A) = σ_max / σ_min.
    High condition number indicates ill-conditioning.
    """
    _, singular_values = np.linalg.svd(A, compute_uv=False)
    condition_number = singular_values[0] / singular_values[-1]
    rank_approx = np.sum(singular_values > 1e-10 * singular_values[0])

    return {
        "condition_number": condition_number,
        "rank_approx": rank_approx,
        "singular_values": singular_values,
        "smallest_singular_value": singular_values[-1],
    }


if __name__ == "__main__":
    # Test case 1: Solve Ax = b
    A = np.array([
        [4.0, 3.0, 2.0],
        [6.0, 3.0, 4.0],
        [3.0, 2.0, 3.0],
    ])
    b = np.array([25.0, 44.0, 23.0])

    x = solve_linear_system(A, b)
    print(f"Solution x = {x}")
    print(f"Verification Ax = {A @ x}")
    print(f"Expected b = {b}\n")

    # Test case 2: Eigenvalue iteration
    A_test = np.array([
        [4.0, -1.0, 0.0],
        [-1.0, 3.0, -1.0],
        [0.0, -1.0, 2.0],
    ])

    lambda_max, eigenvector = power_iteration(A_test)
    print(f"Largest eigenvalue ≈ {lambda_max:.6f}")
    print(f"Eigenvector: {eigenvector}\n")

    # Test case 3: QR decomposition
    A_qr = np.random.randn(5, 3)
    Q, R = qr_decomposition(A_qr)
    print(f"QR reconstruction error: {np.linalg.norm(A_qr - Q @ R):.2e}")
    print(f"Orthogonality check (Q^T Q - I): {np.linalg.norm(Q.T @ Q - np.eye(3)):.2e}\n")

    # Test case 4: SVD truncation
    A_svd = np.random.randn(10, 6)
    U, s, Vt = svd_truncated(A_svd, rank=2)
    A_approx = U @ np.diag(s) @ Vt
    reconstruction_error = np.linalg.norm(A_svd - A_approx)
    print(f"Truncated SVD (rank=2) reconstruction error: {reconstruction_error:.4f}\n")

    # Test case 5: Conditioning analysis
    ill_conditioned = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0000001, 1.0],
        [1.0, 1.0, 1.0000001],
    ])

    analysis = condition_number_analysis(ill_conditioned)
    print(f"Condition number: {analysis['condition_number']:.2e}")
    print(f"Approximate rank: {analysis['rank_approx']}")
    print(f"Singular values: {analysis['singular_values']}")
