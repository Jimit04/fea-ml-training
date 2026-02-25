"""Adjacency matrix construction for structured FEA meshes."""

import numpy as np


def build_beam_adjacency(nx=21, ny=6, nz=6):
    """Build normalised adjacency matrix for a structured hex mesh.

    Computes ``A_hat = D^{-1/2} (A + I) D^{-1/2}`` where nodes are
    adjacent if they differ by 1 step along any single axis.

    Parameters
    ----------
    nx, ny, nz : int, optional
        Number of nodes along each axis (default 21, 6, 6).

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetrically normalised adjacency matrix as ``float32``,
        where ``N = nx * ny * nz``.
    """
    N = nx * ny * nz

    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    rows, cols = [], []
    # Self-loops are already included (A + I)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = idx(i, j, k)
                rows.append(n); cols.append(n)  # self-loop
                for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        rows.append(n); cols.append(idx(ni, nj, nk))

    A = np.zeros((N, N), dtype=np.float32)
    A[rows, cols] = 1.0

    # Symmetric normalisation: A_hat = D^{-1/2} A D^{-1/2}
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.astype(np.float32)             # (N, N)
