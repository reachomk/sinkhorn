import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import matplotlib as mpl
import os

# ============================================================
# Configuration
# ============================================================
N = 100
eta = 0.1
n_steps = 2000
sinkhorn_iters = 100
seed = 42
eps_values = [10.0, 1.0, 0.1, 0.01]

np.random.seed(seed)

# ============================================================
# Source: single 2D Gaussian
# ============================================================
source_mean = np.array([0.0, 0.0])
source_cov = np.eye(2) * 1.5
x_source = np.random.multivariate_normal(source_mean, source_cov, N)

# ============================================================
# Target: mixture of 4 Gaussians at corners
# ============================================================
target_means = [
    np.array([ 3.0,  3.0]),
    np.array([ 3.0, -3.0]),
    np.array([-3.0,  3.0]),
    np.array([-3.0, -3.0]),
]
target_std = 0.5
target_cov = np.eye(2) * target_std**2

y_target = np.zeros((N, 2))
assignments = np.random.choice(4, size=N)
for i in range(N):
    y_target[i] = np.random.multivariate_normal(
        target_means[assignments[i]], target_cov
    )

# ============================================================
# Target density for contour plot
# ============================================================
grid_lim = 6.0
grid_n = 200
gx = np.linspace(-grid_lim, grid_lim, grid_n)
gy = np.linspace(-grid_lim, grid_lim, grid_n)
GX, GY = np.meshgrid(gx, gy)
pos = np.stack([GX, GY], axis=-1)

density = np.zeros((grid_n, grid_n))
for m in target_means:
    rv = multivariate_normal(mean=m, cov=target_cov)
    density += 0.25 * rv.pdf(pos)

# ============================================================
# Helper functions
# ============================================================
def pairwise_l2(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return (diff**2).sum(axis=-1)

def compute_plan_onesided(x, y, eps, mask_diag=False):
    dist = pairwise_l2(x, y)
    if mask_diag and dist.shape[0] == dist.shape[1]:
        np.fill_diagonal(dist, 1e6)
    logits = -dist / eps
    log_T = logits - logsumexp(logits, axis=1, keepdims=True)
    return np.exp(log_T)

def compute_plan_twosided(x, y, eps, mask_diag=False):
    dist = pairwise_l2(x, y)
    if mask_diag and dist.shape[0] == dist.shape[1]:
        np.fill_diagonal(dist, 1e6)
    logits = -dist / eps
    A_row = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    A_col = np.exp(logits - logsumexp(logits, axis=0, keepdims=True))
    return np.sqrt(A_row * A_col)

def compute_plan_sinkhorn(x, y, eps, n_iter=100, mask_diag=False):
    dist = pairwise_l2(x, y)
    if mask_diag and dist.shape[0] == dist.shape[1]:
        np.fill_diagonal(dist, 1e6)
    log_T = -dist / eps
    for _ in range(n_iter):
        log_T = log_T - logsumexp(log_T, axis=1, keepdims=True)
        log_T = log_T - logsumexp(log_T, axis=0, keepdims=True)
    return np.exp(log_T)

def drift_particles(x_init, y_target, plan_fn, eps, eta, n_steps, mask_diag=False, **kwargs):
    N = x_init.shape[0]
    traj = np.zeros((n_steps + 1, N, 2))
    x = x_init.copy()
    traj[0] = x.copy()
    for t in range(n_steps):
        Tp = plan_fn(x, y_target, eps, **kwargs)
        row_sums = np.maximum(Tp.sum(axis=1, keepdims=True), 1e-12)
        bary = (Tp @ y_target) / row_sums
        Vp = bary - x

        Tn = plan_fn(x, x, eps, mask_diag=mask_diag, **kwargs)
        row_sums = np.maximum(Tn.sum(axis=1, keepdims=True), 1e-12)
        bary = (Tn @ x) / row_sums
        Vn = bary - x

        x = x + eta * (Vp - Vn)
        traj[t + 1] = x.copy()
    return traj

# ============================================================
# Methods
# ============================================================
plan_fns = [
    ("One-Sided", compute_plan_onesided, {}),
    ("Two-Sided", compute_plan_twosided, {}),
    ("Sinkhorn", compute_plan_sinkhorn, {"n_iter": sinkhorn_iters}),
]

# ============================================================
# Run all combinations
# ============================================================
results = {}  # (method_idx, eps_idx) -> traj
for mi, (mname, pfn, kw) in enumerate(plan_fns):
    for ei, eps in enumerate(eps_values):
        print(f"  {mname}, eps={eps}")
        use_mask = (mname in ("One-Sided", "Two-Sided"))
        traj = drift_particles(x_source, y_target, pfn, eps, eta, n_steps, mask_diag=use_mask, **kw)
        results[(mi, ei)] = traj

# ============================================================
# Publication-quality 3x4 plot
# ============================================================
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'figure.titlesize': 18,
})

n_rows, n_cols = 3, 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12.5),
                         gridspec_kw={'wspace': 0.04, 'hspace': 0.08})

method_labels = ["One-Sided", "Two-Sided", "Sinkhorn"]
eps_labels = [r"$\varepsilon = 10$", r"$\varepsilon = 1$",
              r"$\varepsilon = 0.1$", r"$\varepsilon = 0.01$"]

for mi in range(n_rows):
    for ei in range(n_cols):
        ax = axes[mi, ei]
        traj = results[(mi, ei)]

        # Target density contours
        ax.contourf(GX, GY, density, levels=20, cmap='Blues', alpha=0.85)
        ax.contour(GX, GY, density, levels=10, colors='steelblue',
                   linewidths=0.4, alpha=0.5)

        # Trajectories
        for i in range(N):
            ax.plot(traj[:, i, 0], traj[:, i, 1],
                    color='red', alpha=0.25, linewidth=0.5)

        # Source (black), final (dark red), target (blue)
        ax.scatter(traj[0, :, 0], traj[0, :, 1],
                   s=8, c='black', zorder=5, edgecolors='none')
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1],
                   s=8, c='darkred', zorder=5, edgecolors='none')
        ax.scatter(y_target[:, 0], y_target[:, 1],
                   s=8, c='blue', zorder=4, alpha=0.35, edgecolors='none')

        ax.set_xlim(-grid_lim, grid_lim)
        ax.set_ylim(-grid_lim, grid_lim)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Column titles (top row only)
        if mi == 0:
            ax.set_title(eps_labels[ei], fontsize=18, fontweight='bold', pad=8)

        # Row labels (left column only)
        if ei == 0:
            ax.set_ylabel(method_labels[mi], fontsize=18, fontweight='bold',
                          labelpad=10)

# Shared legend at bottom
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
           markersize=7, label='Source'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred',
           markersize=7, label='Final'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markersize=7, alpha=0.5, label='Target'),
    Line2D([0], [0], color='red', alpha=0.5, linewidth=1.5,
           label='Trajectory'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=15, frameon=False, bbox_to_anchor=(0.5, -0.01),
           columnspacing=2.0, handletextpad=0.5)

fig.suptitle(
    r'Particle Drift Trajectories with Diagonal Masking ($N$=%d, $\eta$=%.1f, steps=%d)'
    % (N, eta, n_steps),
    fontsize=20, fontweight='bold', y=1.01
)

os.makedirs('/home/hep3/drift/outputs', exist_ok=True)
plt.savefig('/home/hep3/drift/outputs/drift_grid_3x4_diagmask.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.savefig('/home/hep3/drift/outputs/drift_grid_3x4_diagmask.pdf',
            bbox_inches='tight', facecolor='white')
print("Done — saved drift_grid_3x4_diagmask.png and .pdf")
