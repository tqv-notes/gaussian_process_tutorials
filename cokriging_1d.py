"""
ordinary cokriging in 1D with numpy + pytorch

pipeline (it follows the MATLAB version in gaussian_process_tutorials/cokriging):
  1. generate primary (sparse) and secondary (dense) data,
  2. estimate empirical variograms and the cross-variogram,
  3. fit a linear model of coregionalization (LMC) with pytorch,
     using a Cholesky parametrization to guarantee positive semi-definiteness,
  4. solve the ordinary kriging and cokriging systems and compare predictions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)

# data: Z1 = primary (sparse, the one we want to predict), Z2 = secondary
X1 = np.arange(0.0, 0.6 + 1e-9, 0.02) # primary: only on [0, 0.6]
X2 = np.arange(0.0, 1.0 + 1e-9, 0.01) # secondary: covers [0, 1]

Y1 = np.cos(2 * np.pi * X1 + 1.5) + 0.03 * rng.standard_normal(X1.shape)
Y2 = np.cos(2 * np.pi * X2 + 1.0) + 0.03 * rng.standard_normal(X2.shape)


# empirical (cross-)variograms
def empirical_variogram(X, Y, n_bins=20):
    """binned semivariogram: gamma(h) = 0.5 * E[(Z(u+h) - Z(u))^2]."""
    dist = np.abs(X[:, None] - X[None, :]).ravel()
    sqdiff = ((Y[:, None] - Y[None, :]) ** 2).ravel()
    return _bin_variogram(dist, 0.5 * sqdiff, n_bins)


def empirical_cross_variogram(X1, Y1, X2, Y2, n_bins=20):
    """gamma_12(h) = 0.5 * E[(Z1(u+h) - Z1(u)) (Z2(u+h) - Z2(u))].
    Needs both outputs at the same locations -> use collocated points only."""
    i1, i2 = np.nonzero(np.isclose(X1[:, None], X2[None, :]))
    Xc, Y1c, Y2c = X1[i1], Y1[i1], Y2[i2]
    dist = np.abs(Xc[:, None] - Xc[None, :]).ravel()
    cross = ((Y1c[:, None] - Y1c[None, :]) * (Y2c[:, None] - Y2c[None, :])).ravel()
    return _bin_variogram(dist, 0.5 * cross, n_bins)


def _bin_variogram(dist, gam, n_bins):
    # round distances so that pairs with the same nominal lag always fall in
    # the same bin. On a regular grid, float jitter in |x_i - x_j| otherwise
    # splits equal-lag pairs across two adjacent bins whenever a lag coincides
    # with a bin edge - and the two halves can have very different means.
    dist = np.round(dist, 9)
    # only use lags up to half the maximum distance (few pairs beyond that)
    edges = np.linspace(0, dist.max() / 2, n_bins + 1)
    h, g = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (dist >= lo) & (dist < hi) & (dist > 0)
        if sel.sum() > 0:
            h.append(0.5 * (lo + hi))
            g.append(gam[sel].mean())
    return np.array(h), np.array(g)


h11, g11 = empirical_variogram(X1, Y1)                       # primary
h22, g22 = empirical_variogram(X2, Y2)                       # secondary
h12, g12 = empirical_cross_variogram(X1, Y1, X2, Y2)         # cross


#------------------------------------------------------------------------------
# fit a linear model of coregionalization with pytorch
#
#    gamma_ij(h) = B_ij * gamma_m(h; ell) + delta_ij * c0_i
#
#    gamma_m is a Matern-5/2 variogram with unit sill, B = L L^T with L lower
#    triangular, so B is positive semi-definite by construction and the model
#    is valid - no nonlinear constraint needed (cf. fmincon in MATLAB).
#------------------------------------------------------------------------------
def matern52_vario(h, ell):
    """Matern nu=5/2 variogram with unit sill."""
    s = np.sqrt(5.0) * h / ell
    return 1.0 - (1.0 + s + s ** 2 / 3.0) * torch.exp(-s)

def solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]

def lmc_variograms(params, h11, h22, h12):
    raw_L, raw_ell, raw_c0 = params
    ell = torch.nn.functional.softplus(raw_ell)
    c0 = torch.nn.functional.softplus(raw_c0) # nuggets (noise)
    L = torch.tril(raw_L)
    L = L - torch.diag(torch.diag(L)) + torch.diag(torch.nn.functional.softplus(torch.diag(raw_L)))
    B = L @ L.T # coregionalization matrix
    return (B[0, 0] * matern52_vario(h11, ell) + c0[0],
            B[1, 1] * matern52_vario(h22, ell) + c0[1],
            B[0, 1] * matern52_vario(h12, ell)), B, ell, c0

t = lambda a: torch.as_tensor(a, dtype=torch.float64)
th11, tg11, th22, tg22, th12, tg12 = map(t, (h11, g11, h22, g22, h12, g12))

best = {"loss": np.inf}
for restart in range(5):
    torch.manual_seed(restart)
    params = [torch.randn(2, 2, dtype=torch.float64, requires_grad=True),        # raw_L
              (0.5 * torch.rand(1, dtype=torch.float64) - 1.5).requires_grad_(), # raw_ell
              torch.full((2,), -6.0, dtype=torch.float64, requires_grad=True)]   # raw nuggets
    opt = torch.optim.Adam(params, lr=0.01)
    for step in range(3000):
        opt.zero_grad()
        (m11, m22, m12), *_ = lmc_variograms(params, th11, th22, th12)
        loss = ((m11 - tg11)**2).mean() + ((m22 - tg22)**2).mean() + ((m12 - tg12)**2).mean()
        loss.backward()
        opt.step()
    if loss.item() < best["loss"]:
        best = {"loss": loss.item(), "params": [p.detach().clone() for p in params]}

with torch.no_grad():
    _, B, ell, c0 = lmc_variograms(best["params"], th11, th22, th12)
B, ell, c0 = B.numpy(), ell.item(), c0.numpy()
print(f"loss={best['loss']:.3e}  ell={ell:.3f}  nuggets={c0}\nB=\n{B}")
print(f"PSD check: det(B)={np.linalg.det(B):.4f}, B12^2 <= B11*B22: "
      f"{B[0,1]**2 <= B[0,0]*B[1,1]}")

def gamma_model(h, i, j):
    """fitted variogram model gamma_ij evaluated at lag(s) h (numpy)."""
    with torch.no_grad():
        g = float(B[i, j]) * matern52_vario(t(h), t(ell)).numpy()
    return g + (c0[i] if i == j else 0.0) * (np.asarray(h) > 0)

# ordinary kriging of the primary data alone (baseline)
Xt = np.linspace(0.0, 1.0, 200) # prediction grid
n1, n2, nt = len(X1), len(X2), len(Xt)

D11 = np.abs(X1[:, None] - X1[None, :])
A = gamma_model(D11, 0, 0)
A = np.block([[A, np.ones((n1, 1))], [np.ones((1, n1)), np.zeros((1, 1))]])
b = np.vstack([gamma_model(np.abs(X1[:, None] - Xt[None, :]), 0, 0),
               np.ones((1, nt))])
lam = solve(A, b)
Y_krig = lam[:n1].T @ Y1
S_krig = np.sqrt(np.maximum((lam * b).sum(axis=0), 0))

# ordinary cokriging: primary + secondary data jointly
D22 = np.abs(X2[:, None] - X2[None, :])
D12 = np.abs(X1[:, None] - X2[None, :])

G = np.block([[gamma_model(D11, 0, 0), gamma_model(D12, 0, 1)],
              [gamma_model(D12.T, 0, 1), gamma_model(D22, 1, 1)]])
e1 = np.concatenate([np.ones(n1), np.zeros(n2)]) # sum(lambda_1) = 1
e2 = np.concatenate([np.zeros(n1), np.ones(n2)]) # sum(lambda_2) = 0
A = np.block([[G, e1[:, None], e2[:, None]],
              [e1[None, :], np.zeros((1, 2))],
              [e2[None, :], np.zeros((1, 2))]])
b = np.vstack([gamma_model(np.abs(X1[:, None] - Xt[None, :]), 0, 0),
               gamma_model(np.abs(X2[:, None] - Xt[None, :]), 0, 1),
               np.ones((1, nt)),
               np.zeros((1, nt))])
lam = solve(A, b)
Y_cokrig = lam[:n1 + n2].T @ np.concatenate([Y1, Y2])
S_cokrig = np.sqrt(np.maximum((lam * b).sum(axis=0), 0))

# plots and a quick accuracy check on the extrapolation region
truth = np.cos(2 * np.pi * Xt + 1.5)
ext = Xt > 0.6
rmse_k = np.sqrt(np.mean((Y_krig[ext] - truth[ext]) ** 2))
rmse_ck = np.sqrt(np.mean((Y_cokrig[ext] - truth[ext]) ** 2))
print(f"RMSE on x in (0.6, 1.0]:  kriging {rmse_k:.3f}  |  cokriging {rmse_ck:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, (h, g, i, j, name) in zip(axes, [(h11, g11, 0, 0, r"$\gamma_{11}$ (primary)"),
                                         (h22, g22, 1, 1, r"$\gamma_{22}$ (secondary)"),
                                         (h12, g12, 0, 1, r"$\gamma_{12}$ (cross)")]):
    ax.plot(h, g, "o", label="empirical")
    hh = np.linspace(1e-6, h.max(), 200)
    ax.plot(hh, gamma_model(hh, i, j), label="LMC fit")
    ax.set(title=name, xlabel="lag $h$", ylabel=r"$\gamma$")
    ax.legend(frameon=False)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(X2, Y2, "*", color="0.6", label="secondary data")
ax.plot(X1, Y1, "o", label="primary data")
ax.plot(Xt, truth, ":", color="0.4", label="true primary function")
ax.plot(Xt, Y_krig, "-m", lw=1.5, label="kriging (primary only)")
ax.plot(Xt, Y_cokrig, "-k", lw=1.5, label="cokriging")
ax.fill_between(Xt, Y_cokrig - 2 * S_cokrig, Y_cokrig + 2 * S_cokrig,
                color="k", alpha=0.12, label=r"cokriging $\pm 2\sigma$")
ax.set(xlabel="x", ylabel="y")
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
plt.show()