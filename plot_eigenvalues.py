# Plotting of eigenvalues for the channelized coefficient problem.
# Eigenvalues are computed on the fly for contrasts 1 and 1e6, nloc=9.

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.fem import functionspace
from ray.util.multiprocessing import Pool
import ray

import msgfem_parallel as msgfem
import setup

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "text.latex.preamble": r"""
#         \usepackage{amsmath}
#         \usepackage{newtxtext,newtxmath}
#     """
#     })

deg  = 1
Ny   = 4
ny   = 2**8
ol   = 2
os   = 2
nloc = 10
rho  = 0.0
xL, yL, xR, yR = 0, 0, 1, 1
Nx   = Ny          # square domain → Nx = Ny
nx   = ny          # square domain → nx = ny
nDom = Nx * Ny     # 16 subdomains

msh = create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((xL, yL), (xR, yR)),
    n=(nx, ny),
    cell_type=CellType.quadrilateral,
)
V = functionspace(msh, ("Lagrange", deg))
coord_global = V.tabulate_dof_coordinates()


def compute_eigenvalues(contrast, bool_ring):
    """Return absolute eigenvalues (nDom x nloc), sorted descending per row."""
    dirichlet_boundary, robin_boundary, _, _, coeff_A_function = setup.getSetupChannel(
        xL, yL, xR, yR, V, ny, Ny, contrast
    )

    pool = Pool()
    local_data = pool.map(
        msgfem.computeSubdomain,
        [(xR, xL, yR, yL, ol, os, Nx, Ny, nx, ny, nDom,
          coeff_A_function, deg, nloc, i_subdom,
          coord_global, dirichlet_boundary, robin_boundary,
          0.0, rho, bool_ring)
         for i_subdom in range(nDom)]
    )
    pool.close()

    eigenvalues = np.array([np.abs(local_data[i][1]) for i in range(nDom)])
    return np.sort(eigenvalues, axis=1)[:, ::-1]


# --- Compute eigenvalues for the four combinations ---
eigenvalues_true_constant  = compute_eigenvalues(contrast=1,    bool_ring=True)
eigenvalues_false_constant = compute_eigenvalues(contrast=1,    bool_ring=False)
eigenvalues_true_variable  = compute_eigenvalues(contrast=1e6,  bool_ring=True)
eigenvalues_false_variable = compute_eigenvalues(contrast=1e6,  bool_ring=False)

ray.shutdown()


# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
indices = [0, 5, 6]

colors = ["#0072BD", "#D95319", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#77AC30", "#4DBEEE", "#A2142F"]
for ax, i in zip(axes, indices):
    xs = np.arange(1, eigenvalues_true_constant.shape[1])
    ys = np.arange(1, eigenvalues_true_constant.shape[1])
    ax.semilogy(xs, eigenvalues_true_constant[i, ys - 1],  label=r'$R^*$, contrast = 1',          marker='x', linestyle='--', color=colors[0], linewidth=2.5)
    ax.semilogy(xs, eigenvalues_false_constant[i, ys - 1], label=r'$\omega^*$, contrast = 1',      marker='o', linestyle='-',  color=colors[0], linewidth=2.5)
    ax.semilogy(xs, eigenvalues_true_variable[i, ys - 1],  label=r'$R^*$, contrast $= 10^6$',      marker='x', linestyle='--', color=colors[1], linewidth=2.5)
    ax.semilogy(xs, eigenvalues_false_variable[i, ys - 1], label=r'$\omega^*$, contrast $=10^6$',  marker='o', linestyle='-',  color=colors[1], linewidth=2.5)
    ax.set_xlabel('$n$', fontsize=24)
    ax.set_xticks(np.arange(1, eigenvalues_true_constant.shape[1], 1))
    ax.set_xticklabels(np.arange(1, eigenvalues_true_constant.shape[1], 1), fontsize=24)
    ax.grid()

axes[0].set_ylabel(r'$1/\lambda_n$', fontsize=24)
axes[0].legend(loc='best', fontsize=21)
axes[0].set_yticks([10**i for i in range(-2, 9)])
axes[0].set_yticklabels([f'$10^{{{i}}}$' for i in range(-2, 9)], fontsize=24)
axes[0].set_ylim(10**-2, 10**8)
plt.tight_layout()
plt.savefig("plots/eigenvalues_channel_subdomains_os" + str(os) + ".pdf", dpi=600, bbox_inches='tight')
