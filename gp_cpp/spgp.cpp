// ============================================================================
//  spgp.cpp
//  ---------------------------------------------------------------------------
//  a simple, self-contained tutorial implementation of the Sparse Pseudo-input
//  Gaussian Process (SPGP), also known as FITC.
// ============================================================================

#include <armadillo>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace arma;

// ----------------------------------------------------------------------------
// covariance function (squared-exponential / RBF)
// ----------------------------------------------------------------------------
struct SqExpKernel {
  double sf2;   // signal variance  (often called sigma_f^2)
  double ell;   // length-scale

  SqExpKernel(double sf2_ = 1.0, double ell_ = 1.0) : sf2(sf2_), ell(ell_) {}

  // cross-covariance matrix between every column of A (d x nA)
  // and every column of B (d x nB)  ->  (nA x nB).
  mat cross(const mat& A, const mat& B) const {
    mat Kab(A.n_cols, B.n_cols);
    for (uword i = 0; i < A.n_cols; ++i)
      for (uword j = 0; j < B.n_cols; ++j) {
        double d2 = sum(square(A.col(i) - B.col(j)));
        Kab(i, j) = sf2 * std::exp(-d2 / (2.0 * ell * ell));
      }
    return Kab;
  }

  // the prior variance k(x,x) is constant for this kernel.
  double self() const { return sf2; }

  mat sqdist(const mat& A, const mat& B) const {
    mat D(A.n_cols, B.n_cols);
    for (uword i = 0; i < A.n_cols; ++i)
      for (uword j = 0; j < B.n_cols; ++j)
        D(i, j) = sum(square(A.col(i) - B.col(j)));
    return D;
  }
};

// ----------------------------------------------------------------------------
// the SPGP model
// ----------------------------------------------------------------------------
class SPGP {
public:
  SqExpKernel kernel;
  double s2n;      // noise variance
  double jitter;   // small diagonal added to Kmm for numerical stability

  SPGP(const SqExpKernel& k, double noise_var, double jitter_ = 1e-6)
    : kernel(k), s2n(noise_var), jitter(jitter_) {}

  // data + pseudo-inputs
  void set_training(const mat& X_, const rowvec& y_) { X = X_; y = y_; }
  void set_pseudo_inputs(const mat& Xb_)             { Xb = Xb_; }

  // the core factorization
  void fit() {
    const uword M = Xb.n_cols;
    const uword N = X.n_cols;

    mat Kmm = kernel.cross(Xb, Xb) + jitter * eye<mat>(M, M);
    Kmn     = kernel.cross(Xb, X); // M x N (kept for gradient)

    L = chol(Kmm, "lower"); // L L^T = Kmm
    V = solve(trimatl(L), Kmn); // V = L^{-1} Kmn  (unscaled)

    // per-point variance correction
    ep = 1.0 + (kernel.self() - sum(V % V, 0).t()) / s2n; // N x 1

    // scale V in place by 1/sqrt(ep)
    V.each_row() /= sqrt(ep.t()); // now SCALED

    // second Cholesky on the scaled V
    Lm = chol(s2n * eye<mat>(M, M) + V * V.t(), "lower");

    // bet = Lm⁻¹ ( V * (y/sqrt(ep))ᵀ )
    colvec yhat = (y.t()) / sqrt(ep); // N x 1
    bet = solve(trimatl(Lm), V * yhat); // M x 1
  }

  // negative log marginal likelihood
  double neg_log_likelihood() {
    fit();
    const uword N = X.n_cols;
    const uword M = Xb.n_cols;

    // negative log marginal likelihood of the FITC/SPGP model.
    colvec yhat = (y.t()) / sqrt(ep); // N x 1

    double log_det = (double(N) - double(M)) * std::log(s2n)
                   + accu(log(ep))
                   + 2.0 * accu(log(Lm.diag()));

    double quad    = ( dot(yhat, yhat) - dot(bet, bet) ) / s2n;

    double nll = 0.5 * ( log_det + quad + double(N) * std::log(2.0 * datum::pi) );
    return nll;
  }

  // predictive mean and variance at test inputs Xs (d x T)
  void predict(const mat& Xs, rowvec& mu, rowvec& var) {
    fit();
    mat Ksm = kernel.cross(Xb, Xs);                 // M x T
    mat lst = solve(trimatl(L),  Ksm);              // M x T
    mat lmst = solve(trimatl(Lm), lst);             // M x T

    mu = (bet.t() * lmst);                          // 1 x T  (zero mean added = 0)

    rowvec s1 = sum(lst  % lst,  0);                // 1 x T
    rowvec s2 = sum(lmst % lmst, 0);                // 1 x T
    var = kernel.self() - s1 + s2n * s2 + s2n;      // predictive variance (+noise)
  }

  // analytic gradient of NLL wrt each pseudo-input coordinate
  mat grad_pseudo_inputs_fd(double h = 1e-4) {
    mat G(Xb.n_rows, Xb.n_cols, fill::zeros);
    for (uword i = 0; i < Xb.n_rows; ++i)
      for (uword j = 0; j < Xb.n_cols; ++j) {
        double orig = Xb(i, j);
        Xb(i, j) = orig + h; double fp = neg_log_likelihood();
        Xb(i, j) = orig - h; double fm = neg_log_likelihood();
        Xb(i, j) = orig;
        G(i, j) = (fp - fm) / (2.0 * h);
      }
    return G;
  }

  // gradient of NLL wrt the (positive) hyperparameters sf2, ell, s2n
  vec grad_hyper_log_fd(double h = 1e-4) {
    vec g(3, fill::zeros);

    // sf2
    { double o = kernel.sf2;
      kernel.sf2 = o * std::exp(+h); double fp = neg_log_likelihood();
      kernel.sf2 = o * std::exp(-h); double fm = neg_log_likelihood();
      kernel.sf2 = o; g(0) = (fp - fm) / (2.0 * h); }
    // ell
    { double o = kernel.ell;
      kernel.ell = o * std::exp(+h); double fp = neg_log_likelihood();
      kernel.ell = o * std::exp(-h); double fm = neg_log_likelihood();
      kernel.ell = o; g(1) = (fp - fm) / (2.0 * h); }
    // s2n
    { double o = s2n;
      s2n = o * std::exp(+h); double fp = neg_log_likelihood();
      s2n = o * std::exp(-h); double fm = neg_log_likelihood();
      s2n = o; g(2) = (fp - fm) / (2.0 * h); }

    return g;
  }

  // apply a log-space step to the hyperparameters (keeps them positive).
  void step_hyper_log(const vec& dlog) {
    kernel.sf2 *= std::exp(dlog(0));
    kernel.ell *= std::exp(dlog(1));
    s2n        *= std::exp(dlog(2));
  }

  const mat& pseudo_inputs() const { return Xb; }
  mat&       pseudo_inputs()       { return Xb; }

  // ==========================================================================
  // exact analytic gradients via direct differentiation of the FITC covariance
  //
  //   Sigma = s2n I + diag(lam) + Phi^T Phi ,   Phi = L^-1 Kmn  (M x N)
  //   lam_i = kxx - ||Phi_i||^2
  //
  // for any parameter theta, the NLL gradient is the standard Gaussian form
  //   dNLL/dtheta = 0.5 tr[ (Sigma^-1 - b b^T) dSigma/dtheta ],   b = Sigma^-1 y
  //
  // NOTE: this forms N x N matrices and inverts Sigma, so it is O(N^3) once
  // plus O(N^2) per parameter. the optimal solution should be O(N M^2).
  // Snelson's implementation avoids the N x N inverse and is faster for large N,
  // but it is also far harder to read and to verify. for a tutorial we choose
  // the transparent exact form. for production with large N, prefer the
  // Woodbury-based O(N M^2) gradient. either way, verify against finite
  // differences (as the demo does) before trusting a hand-derived version.
  // ==========================================================================
private:
  // assemble Sigma and the reusable factor Gmat = Sigma^-1 - b b^T.
  void build_sigma_factor(mat& Sigma, mat& Sinv, mat& Gmat, colvec& Phi_self,
                          mat& Phi, mat& Kmn_out) {
    const uword N = X.n_cols;
    const uword M = Xb.n_cols;

    mat Kmm = kernel.cross(Xb, Xb) + jitter * eye<mat>(M, M);
    Kmn_out = kernel.cross(Xb, X);                       // M x N (unscaled)
    mat Lc  = chol(Kmm, "lower");
    Phi     = solve(trimatl(Lc), Kmn_out);              // M x N

    colvec lam = kernel.self() - sum(Phi % Phi, 0).t(); // N x 1
    Sigma = Phi.t() * Phi;                               // N x N
    Sigma.diag() += lam + s2n;                           // add diag(lam)+s2n I

    Sinv = inv_sympd(Sigma);
    colvec b = Sinv * y.t();
    Gmat = Sinv - b * b.t();                             // N x N symmetric
    Phi_self = lam;                                      // (unused placeholder)
  }

public:
  // analytic log-space gradient wrt sf2, ell, s2n
  vec grad_hyper_log_analytic() {
    const uword N = X.n_cols, M = Xb.n_cols;
    mat Sigma, Sinv, Gmat, Phi, KmnU; colvec lam;
    build_sigma_factor(Sigma, Sinv, Gmat, lam, Phi, KmnU);

    mat Kmm = kernel.cross(Xb, Xb) + jitter * eye<mat>(M, M);
    mat Knn = kernel.cross(X, X);                        // for d/dsf2, d/dell
    mat Dnn = kernel.sqdist(X, X);
    mat Dmm = kernel.sqdist(Xb, Xb);
    mat Dmn = kernel.sqdist(Xb, X);

    // we need dSigma for each theta. Sigma = Knn_fitc; differentiate the FITC
    // form Sigma = diag(kxx - diag(Qnn)) + Qnn + s2n I, Qnn = Knm Kmm^-1 Kmn.
    // it is cleaner to differentiate Qnn and the diagonal correction together:
    //   Qnn = Knm Kmm^-1 Kmn
    // dQnn = dKnm Kmm^-1 Kmn - Knm Kmm^-1 dKmm Kmm^-1 Kmn + Knm Kmm^-1 dKmn
    // Sigma = Qnn + diag(kxx - diag(Qnn)) + s2n I
    // dSigma = dQnn + diag(dkxx - diag(dQnn)) [+ dS2n*I]
    mat KmmInv = inv_sympd(Kmm);
    mat Knm = KmnU.t();                                  // N x M
    mat A   = Knm * KmmInv;                              // N x M  (= Knm Kmm^-1)

    auto dSigma_from = [&](const mat& dKnn_diagsrc, const mat& dKmm,
                           const mat& dKmn, double dkxx)->mat {
      // dQnn
      mat dKnm = dKmn.t();
      mat dQ = dKnm * KmmInv * KmnU
             - A * dKmm * KmmInv * KmnU
             + A * dKmn;                                 // N x N
      mat dS = dQ;
      // diagonal correction: Sigma_ii has (kxx - Qnn_ii) replacing Qnn_ii,
      // i.e. add (dkxx - dQ_ii) on the diagonal.
      dS.diag() += (dkxx - dQ.diag());
      return dS;
    };

    // dsf2: every kernel block scales as (block)/sf2; dkxx = kxx/sf2 = self/sf2
    mat dKnn_sf2 = Knn / kernel.sf2;
    mat dKmm_sf2 = Kmm; dKmm_sf2.diag() -= jitter; dKmm_sf2 /= kernel.sf2;
    mat dKmn_sf2 = KmnU / kernel.sf2;
    double dkxx_sf2 = kernel.self() / kernel.sf2;
    mat dS_sf2 = dSigma_from(dKnn_sf2, dKmm_sf2, dKmn_sf2, dkxx_sf2);

    // dell: dk = k * r^2/ell^3 (elementwise); dkxx = 0 (r=0)
    double e3 = kernel.ell*kernel.ell*kernel.ell;
    mat dKmm_ell = (Kmm); dKmm_ell.diag() -= jitter; dKmm_ell %= (Dmm / e3);
    mat dKmn_ell = KmnU % (Dmn / e3);
    double dkxx_ell = 0.0;
    mat dS_ell = dSigma_from(Knn, dKmm_ell, dKmn_ell, dkxx_ell);

    // ds2n: dSigma = I
    // gradients: 0.5 tr[G dSigma]
    double g_sf2 = 0.5 * accu(Gmat % dS_sf2);
    double g_ell = 0.5 * accu(Gmat % dS_ell);
    double g_s2n = 0.5 * trace(Gmat);

    vec g(3);
    g(0) = kernel.sf2 * g_sf2;
    g(1) = kernel.ell * g_ell;
    g(2) = s2n        * g_s2n;
    return g;
  }

  // analytic gradient wrt pseudo-input locations Xb (d x M)
  mat grad_pseudo_inputs_analytic() {
    const uword N = X.n_cols, M = Xb.n_cols, d = Xb.n_rows;
    mat Sigma, Sinv, Gmat, Phi, KmnU; colvec lam;
    build_sigma_factor(Sigma, Sinv, Gmat, lam, Phi, KmnU);

    mat Kmm = kernel.cross(Xb, Xb) + jitter * eye<mat>(M, M);
    mat KmmInv = inv_sympd(Kmm);
    mat KmnRaw = KmnU;                                   // M x N
    mat Knm = KmnRaw.t();                                // N x M
    mat A   = Knm * KmmInv;                              // N x M

    // precompute kernel value matrices (no-jitter) for derivative factors.
    mat Kmm0 = kernel.cross(Xb, Xb);                    // M x M
    double inv_e2 = 1.0 / (kernel.ell * kernel.ell);

    mat G_out(d, M, fill::zeros);

    // for each pseudo-input a and coordinate i, dKmm and dKmn are sparse
    // (only row/col a nonzero). We exploit that for O(N^2) per (i,a) instead
    // of O(N^3).  dk(x_a, z)/dx_a,i = k(x_a,z) * (z_i - x_a,i)/ell^2
    for (uword a = 0; a < M; ++a) {
      // kernel rows for pseudo-input a
      rowvec kmn_a = KmnRaw.row(a);                      // 1 x N  k(x_a, X_j)
      rowvec kmm_a = Kmm0.row(a);                        // 1 x M  k(x_a, X_b)

      for (uword i = 0; i < d; ++i) {
        // derivative factors (z_i - x_a,i)
        rowvec dx_mn = (X.row(i) - Xb(i,a));             // 1 x N
        rowvec dx_mm = (Xb.row(i) - Xb(i,a));            // 1 x M

        rowvec dKmn_a = kmn_a % dx_mn * inv_e2;          // 1 x N
        rowvec dKmm_a = kmm_a % dx_mm * inv_e2;          // 1 x M (a-th row of dKmm)

        // build dQnn cheaply. dKmn has only row a nonzero (= dKmn_a).
        // dKmm has row a and col a nonzero (symmetric), diag a = 0.
        // dQ = dKnm Kmm^-1 Kmn  -  A dKmm Kmm^-1 Kmn  +  A dKmn
        //
        // Let u = A.col(a) (N x 1) = (Knm Kmm^-1)[:,a].
        colvec u = A.col(a);                             // N x 1
        // term3: A dKmn -> only column-structure: A * dKmn where dKmn row a only
        //   (A dKmn)_pq = A_pa * dKmn_a,q  = u_p * dKmn_a(q)
        mat term3 = u * dKmn_a;                          // N x N
        // term1: dKnm Kmm^-1 Kmn ; dKnm col a only = (dKmn_a)^T at col a
        //   dKnm Kmm^-1 Kmn = dKmn_a^T (row a of Kmm^-1 Kmn).
        rowvec KmmInvKmn_a = KmmInv.row(a) * KmnRaw;     // 1 x N
        mat term1 = dKmn_a.t() * KmmInvKmn_a;            // N x N
        // term2: A dKmm Kmm^-1 Kmn. dKmm = e_a dKmm_a^T + dKmm_a e_a^T (sym, diag0)
        //   A dKmm = A(:,a) dKmm_a^T + (A dKmm_a) e_a^T
        colvec AdKmm_a = A * dKmm_a.t();                 // N x 1
        mat AdKmm = u * dKmm_a + AdKmm_a * unit_e(a, M).t();  // N x M
        mat term2 = (AdKmm * KmmInv) * KmnRaw;           // N x N

        mat dQ = term1 - term2 + term3;
        mat dS = dQ;
        dS.diag() += (0.0 - dQ.diag());                  // dkxx = 0

        G_out(i, a) = 0.5 * accu(Gmat % dS);
      }
    }
    return G_out;
  }

private:
  static colvec unit_e(uword k, uword n) { colvec e(n, fill::zeros); e(k)=1.0; return e; }
public:

private:
  mat X, Xb;
  rowvec y;
  // cached factors
  mat L, V, Lm, Kmn;     // V is scaled after fit(); Kmn is unscaled M x N
  colvec ep, bet;
};

// ----------------------------------------------------------------------------
// demo
// ----------------------------------------------------------------------------
static void dump_csv(const std::string& path, const mat& Xs,
                     const rowvec& mu, const rowvec& var) {
  std::ofstream f(path);
  for (uword i = 0; i < Xs.n_cols; ++i)
    f << Xs(0, i) << "," << mu(i) << "," << var(i) << "\n";
}

int main() {
  arma_rng::set_seed(1);

  // synthetic data: y = sin(x+x^2/2) + noise on [0, 2pi]
  const uword N = 200;
  mat    X(1, N);
  rowvec y(N);
  for (uword i = 0; i < N; ++i) {
    double xi = 2.0 * datum::pi * double(i) / double(N);
    X(0, i) = xi;
    y(i)    = std::sin(xi + xi * xi / 2.0) + 0.05 * randn();
  }

  // model: SqExp kernel, M = 20 pseudo-inputs
  // start hyperparameters deliberately OFF (too-short length-scale, too-large
  // noise) so the joint optimizer visibly corrects them.
  SqExpKernel kernel(/*sf2=*/0.3, /*ell=*/0.3);
  SPGP gp(kernel, /*noise_var=*/0.1);
  gp.set_training(X, y);

  const uword M = 20;
  /*
  mat Xb(1, M);
  for (uword j = 0; j < M; ++j) // start them clustered low
    Xb(0, j) = 1.0 * datum::pi * double(j) / double(M);
  */
  double lo = X.min(), hi = X.max();
  mat Xb = lo + (hi - lo) * randu<mat>(1, M);
  gp.set_pseudo_inputs(Xb);

  // save the initial pseudo-input locations too
  std::ofstream pt("spgp_pseudo_inputs_ini.csv");
  for (uword j = 0; j < gp.pseudo_inputs().n_cols; ++j)
    pt << gp.pseudo_inputs()(0, j) << ",0\n";

  // test grid
  const uword T = 400;
  mat Xs(1, T);
  for (uword i = 0; i < T; ++i)
    Xs(0, i) = 2.0 * datum::pi * double(i) / double(T);

  rowvec mu, var;
  gp.predict(Xs, mu, var);
  dump_csv("spgp_pred_before.csv", Xs, mu, var);

  // sanity check: analytic gradients vs finite differences
  {
    mat Ga = gp.grad_pseudo_inputs_analytic();
    mat Gn = gp.grad_pseudo_inputs_fd();
    double rel = norm(Ga - Gn, "fro") / (norm(Gn, "fro") + 1e-12);
    std::cout << "pseudo-input gradient check: max|diff| = "
              << abs(Ga - Gn).max()
              << ", relative error = " << rel << "\n";

    vec ha = gp.grad_hyper_log_analytic();
    vec hn = gp.grad_hyper_log_fd();
    std::cout << "hyperparam   gradient check: max|diff| = "
              << max(abs(ha - hn))
              << ", relative error = "
              << norm(ha - hn) / (norm(hn) + 1e-12) << "\n\n";
  }

  // joint optimization: pseudo-inputs + kernel hyperparameters + noise
  const int    iters    = 500;
  double       lr_pi     = 0.02;   // pseudo-input step
  double       lr_hyp    = 0.005;  // hyperparameter (log-space) step

  std::cout << "Initial  NLL = " << gp.neg_log_likelihood()
            << "   [sf2=" << gp.kernel.sf2
            << ", ell=" << gp.kernel.ell
            << ", s2n=" << gp.s2n << "]\n";

  double f_prev = gp.neg_log_likelihood();

  for (int it = 0; it < iters; ++it) {
    // snapshot current state so we can roll back a bad step
    mat   Xb_save  = gp.pseudo_inputs();
    double sf2_s = gp.kernel.sf2, ell_s = gp.kernel.ell, s2n_s = gp.s2n;

    // gradients at current point (ANALYTIC, verified above)
    mat Gpi = gp.grad_pseudo_inputs_analytic();
    vec Ghy = gp.grad_hyper_log_analytic();

    // trial step
    gp.pseudo_inputs() -= lr_pi * Gpi;
    gp.step_hyper_log(-lr_hyp * Ghy);

    double f_new = gp.neg_log_likelihood();

    if (f_new <= f_prev) {
      // accept; gently grow the steps again
      f_prev = f_new;
      lr_pi  *= 1.05;
      lr_hyp *= 1.05;
    } else {
      // reject: roll back and shrink the steps
      gp.pseudo_inputs() = Xb_save;
      gp.kernel.sf2 = sf2_s; gp.kernel.ell = ell_s; gp.s2n = s2n_s;
      lr_pi  *= 0.5;
      lr_hyp *= 0.5;
    }

    if ((it + 1) % 20 == 0)
      std::cout << "  iter " << (it + 1)
                << "  NLL = " << f_prev
                << "   [sf2=" << gp.kernel.sf2
                << ", ell=" << gp.kernel.ell
                << ", s2n=" << gp.s2n << "]\n";
  }

  gp.predict(Xs, mu, var);
  dump_csv("spgp_pred_after.csv", Xs, mu, var);
  std::cout << "Final    NLL = " << gp.neg_log_likelihood()
            << "   [sf2=" << gp.kernel.sf2
            << ", ell=" << gp.kernel.ell
            << ", s2n=" << gp.s2n << "]\n";

  // save the learned pseudo-input locations too
  std::ofstream pf("spgp_pseudo_inputs.csv");
  for (uword j = 0; j < gp.pseudo_inputs().n_cols; ++j)
    pf << gp.pseudo_inputs()(0, j) << ",0\n";

  std::cout << "wrote spgp_pred_before.csv, spgp_pred_after.csv, "
            << "spgp_pseudo_inputs_ini.csv, spgp_pseudo_inputs.csv\n";
            
  return 0;
}
