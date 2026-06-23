// ============================================================================
//  gp.cpp
//  ---------------------------------------------------------------------------
//  a simple, self-contained tutorial implementation of standard (full) Gaussian
//  Process regression. Written as the companion / baseline to the SPGP (FITC)
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
  double sf2;   // signal variance
  double ell;   // length-scale

  SqExpKernel(double sf2_ = 1.0, double ell_ = 1.0) : sf2(sf2_), ell(ell_) {}

  mat cross(const mat& A, const mat& B) const {
    mat Kab(A.n_cols, B.n_cols);
    for (uword i = 0; i < A.n_cols; ++i)
      for (uword j = 0; j < B.n_cols; ++j) {
        double d2 = sum(square(A.col(i) - B.col(j)));
        Kab(i, j) = sf2 * std::exp(-d2 / (2.0 * ell * ell));
      }
    return Kab;
  }

  // squared-distance matrix r^2_ij = ||A_i - B_j||^2, needed by the
  // length-scale derivative. Returned alongside cross() to avoid recomputing.
  mat sqdist(const mat& A, const mat& B) const {
    mat D(A.n_cols, B.n_cols);
    for (uword i = 0; i < A.n_cols; ++i)
      for (uword j = 0; j < B.n_cols; ++j)
        D(i, j) = sum(square(A.col(i) - B.col(j)));
    return D;
  }

  // element-wise partial derivatives of the kernel matrix K = cross(A,B).
  mat dK_dsf2(const mat& K) const { return K / sf2; }
  mat dK_dell(const mat& K, const mat& D) const {
    return K % (D / (ell * ell * ell));
  }

  double self() const { return sf2; }
};

// ----------------------------------------------------------------------------
// the full GP
// ----------------------------------------------------------------------------
class GP {
public:
  SqExpKernel kernel;
  double s2n;      // noise variance
  double jitter;   // tiny diagonal for numerical safety

  GP(const SqExpKernel& k, double noise_var, double jitter_ = 1e-9)
    : kernel(k), s2n(noise_var), jitter(jitter_) {}

  void set_training(const mat& X_, const rowvec& y_) { X = X_; y = y_; }

  // the core factorization
  void fit() {
    const uword N = X.n_cols;
    mat Ky = kernel.cross(X, X) + (s2n + jitter) * eye<mat>(N, N);

    // L L^T = Ky
    L = chol(Ky, "lower");

    // alpha = Ky^-1 y  via two triangular solves
    colvec yc = y.t();                          // N x 1
    colvec v  = solve(trimatl(L),       yc);    // L v = y
    alpha     = solve(trimatu(L.t()),   v);     // L^T alpha = v
  }

  // negative log marginal likelihood
  double neg_log_likelihood() {
    fit();
    const uword N = X.n_cols;
    double quad    = 0.5 * dot(y.t(), alpha);
    double logdet  = accu(log(L.diag()));                 // 0.5 log|Ky|
    return quad + logdet + 0.5 * double(N) * std::log(2.0 * datum::pi);
  }

  // predictive mean and variance at test inputs Xs (d x T)
  void predict(const mat& Xs, rowvec& mu, rowvec& var) {
    fit();
    mat Ks = kernel.cross(X, Xs);               // N x T
    mu = (Ks.t() * alpha).t();                  // 1 x T

    mat v = solve(trimatl(L), Ks);              // N x T
    rowvec s = sum(v % v, 0);                    // 1 x T
    var = kernel.self() + s2n - s;               // predictive variance (+noise)
  }

  // analytic log-space gradient of NLL wrt sf2, ell, s2n
  vec grad_hyper_log_analytic() {
    fit();                                    // ensures L, alpha current
    const uword N = X.n_cols;

    // Ky^-1 from the Cholesky: solve Ky * Kinv = I using L.
    mat Kinv = solve(trimatu(L.t()), solve(trimatl(L), eye<mat>(N, N)));

    mat AA  = alpha * alpha.t();              // N x N
    mat M   = Kinv - AA;                      // the (Ky^-1 - alpha alpha^T) factor

    mat K   = kernel.cross(X, X);             // noise-free part (for sf2, ell)
    mat D   = kernel.sqdist(X, X);

    mat dK_sf2 = kernel.dK_dsf2(K);           // dKy/d sf2
    mat dK_ell = kernel.dK_dell(K, D);        // dKy/d ell
    // dKy/d s2n = I

    double g_sf2 = 0.5 * accu(M % dK_sf2);
    double g_ell = 0.5 * accu(M % dK_ell);
    double g_s2n = 0.5 * trace(M);            // tr[M * I]

    // chain rule to log-space: d/d(log theta) = theta * d/d theta
    vec g(3);
    g(0) = kernel.sf2 * g_sf2;
    g(1) = kernel.ell * g_ell;
    g(2) = s2n        * g_s2n;
    return g;
  }

  // finite-difference gradient (kept only as a correctness oracle)
  vec grad_hyper_log_fd(double h = 1e-4) {
    vec g(3, fill::zeros);
    { double o = kernel.sf2;
      kernel.sf2 = o*std::exp(+h); double fp = neg_log_likelihood();
      kernel.sf2 = o*std::exp(-h); double fm = neg_log_likelihood();
      kernel.sf2 = o; g(0) = (fp - fm)/(2*h); }
    { double o = kernel.ell;
      kernel.ell = o*std::exp(+h); double fp = neg_log_likelihood();
      kernel.ell = o*std::exp(-h); double fm = neg_log_likelihood();
      kernel.ell = o; g(1) = (fp - fm)/(2*h); }
    { double o = s2n;
      s2n = o*std::exp(+h); double fp = neg_log_likelihood();
      s2n = o*std::exp(-h); double fm = neg_log_likelihood();
      s2n = o; g(2) = (fp - fm)/(2*h); }
    return g;
  }

  void step_hyper_log(const vec& dlog) {
    kernel.sf2 *= std::exp(dlog(0));
    kernel.ell *= std::exp(dlog(1));
    s2n        *= std::exp(dlog(2));
  }

private:
  mat X;
  rowvec y;
  mat    L;       // lower Cholesky of Ky
  colvec alpha;   // Ky^-1 y
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

  // model: start hyperparameters deliberately OFF
  SqExpKernel kernel(/*sf2=*/0.3, /*ell=*/0.3);
  GP gp(kernel, /*noise_var=*/0.1);
  gp.set_training(X, y);

  // test grid
  const uword T = 400;
  mat Xs(1, T);
  for (uword i = 0; i < T; ++i)
    Xs(0, i) = 2.0 * datum::pi * double(i) / double(T);

  rowvec mu, var;
  gp.predict(Xs, mu, var);
  dump_csv("gp_pred_before.csv", Xs, mu, var);

  // sanity check: analytic gradient vs finite differences
  {
    vec ga = gp.grad_hyper_log_analytic();
    vec gn = gp.grad_hyper_log_fd();
    std::cout << "gradient check (log-space):\n";
    std::cout << "  analytic = " << ga.t();
    std::cout << "  fin-diff = " << gn.t();
    std::cout << "  max|diff| = " << max(abs(ga - gn)) << "\n\n";
  }

  // learn hyperparameters: loss-checked log-space gradient descent
  const int iters = 120;
  double lr = 0.01;

  std::cout << "Initial  NLL = " << gp.neg_log_likelihood()
            << "   [sf2=" << gp.kernel.sf2
            << ", ell=" << gp.kernel.ell
            << ", s2n=" << gp.s2n << "]\n";

  double f_prev = gp.neg_log_likelihood();
  for (int it = 0; it < iters; ++it) {
    double sf2_s = gp.kernel.sf2, ell_s = gp.kernel.ell, s2n_s = gp.s2n;

    vec g = gp.grad_hyper_log_analytic();
    gp.step_hyper_log(-lr * g);

    double f_new = gp.neg_log_likelihood();
    if (f_new <= f_prev) { f_prev = f_new; lr *= 1.1; }
    else {
      gp.kernel.sf2 = sf2_s; gp.kernel.ell = ell_s; gp.s2n = s2n_s;
      lr *= 0.5;
    }

    if ((it + 1) % 20 == 0)
      std::cout << "  iter " << (it + 1)
                << "  NLL = " << f_prev
                << "   [sf2=" << gp.kernel.sf2
                << ", ell=" << gp.kernel.ell
                << ", s2n=" << gp.s2n << "]\n";
  }

  gp.predict(Xs, mu, var);
  dump_csv("gp_pred_after.csv", Xs, mu, var);
  std::cout << "Final    NLL = " << gp.neg_log_likelihood()
            << "   [sf2=" << gp.kernel.sf2
            << ", ell=" << gp.kernel.ell
            << ", s2n=" << gp.s2n << "]\n";

  // also dump the noisy training points for plotting
  std::ofstream tf("gp_training.csv");
  for (uword i = 0; i < N; ++i) tf << X(0, i) << "," << y(i) << "\n";

  std::cout << "wrote gp_pred_before.csv, gp_pred_after.csv, gp_training.csv\n";
  return 0;
}
