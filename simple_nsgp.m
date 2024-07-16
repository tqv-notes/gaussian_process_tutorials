%% simple implementation of Non-Stationary Gaussian Process (NSGP) regression

% non-stationary Gaussian kernel
kernel = @(x1,x2,l1,l2,s1,s2) (s1.*s2') .* sqrt((2*l1.*l2')./(l1.^2+l2'.^2)) .* exp( -(sum(x1.^2,2)+sum(x2.^2,2)'-2*x1.*x2')./(l1.^2+l2'.^2) );

x = (-5:0.2:5)';
mu = zeros(size(x));
l = 1*ones(size(x));
s = 1*ones(size(x));
cov = kernel(x,x,l,l,s,s);

samples = mvnrnd(mu,cov,3);

figure();
plot(samples');
title('samples from a non-stationary gaussian process');
axis('tight');

%% GP with default hyperparameters

% stationary Gaussian kernel
kernel_s = @(x1, x2, l, sigma_f) sigma_f^2 * exp( -0.5*( (sum(x1.^2,2)) + (sum(x2.^2,2)).' - 2*x1.*(x2.') ) );

% training data
n = 24;
x_train = linspace(0,1,n)'*10 - 5;
y_train = test_func( (x_train+5)/10 ) + 0.0 * randn(size(x_train));

l = 0.5*ones(size(x_train));
s = 2*ones(size(x_train));
o = 1e-3 * ones(size(x_train));

Kl = kernel_s(x_train, x_train, 0.5, 1) + 1e-3 * eye(length(x_train));
Kl_t = kernel_s(x_train, x, 0.5, 1);
lt = mean(l) + Kl_t' * (inv(Kl) *(l-mean(l)));

Ks = kernel_s(x_train, x_train, 0.5, 1) + 1e-3 * eye(length(x_train));
Ks_t = kernel_s(x_train, x, 0.5, 1);
st = mean(s) + Ks_t' * (inv(Ks) *(s-mean(s)));

Ko = kernel_s(x_train, x_train, 0.5, 1) + 1e-3 * eye(length(x_train));
Ko_t = kernel_s(x_train, x, 0.5, 1);
ot = mean(o) + Ko_t' * (inv(Ko) *(o-mean(o)));

K = kernel(x_train, x_train, l, l, s, s) + diag(o);
K_s = kernel(x_train, x, l, lt, s, st);
K_ss = kernel(x, x, lt, lt, st, st) + diag(ot);

K_inv = inv(K);
mu_s = K_s' * K_inv * y_train;
cov_s = K_ss - K_s' * K_inv * K_s;

lb = mu_s - 2*sqrt( diag(cov_s) );
ub = mu_s + 2*sqrt( diag(cov_s) );

figure();
hold on;
fill( [x; flipud(x)], [lb; flipud(ub)], 'y');
plot( x, mu_s, '-r', 'linewidth', 2);
plot( x, test_func((x+5)/10), '-m' );
plot( x_train, y_train, 'o');

%% hyperparameter optimization via negative log likelihood

nll_l = @(p) 0.5 * log(det(Kl)) + 0.5 * p.' * inv(Kl) * p + 0.5 * length(p) * log(2*pi);
nll_s = @(p) 0.5 * log(det(Ks)) + 0.5 * p.' * inv(Ks) * p + 0.5 * length(p) * log(2*pi);
nll_o = @(p) 0.5 * log(det(Ko)) + 0.5 * p.' * inv(Ko) * p + 0.5 * length(p) * log(2*pi);

% convention: p = [l s o]
% Ker = @(p) kernel(x_train, x_train, p(1:n), p(1:n), p(n+1:2*n), p(n+1:2*n)) + diag(p(2*n+1:end));
Ker = @(p) kernel(x_train, x_train, p(1:n), p(1:n), p(n+1:2*n), p(n+1:2*n)) + diag(o);
nll_y = @(p) 0.5 * log(det(Ker(p))) + 0.5 * y_train.' * inv(Ker(p)) * y_train + 0.5 * length(x_train) * log(2*pi);

% nll = @(p) nll_y(p) + nll_l(p(1:n)) + nll_s(p(n+1:2*n)) + nll_o(p(2*n+1:end));
nll = @(p) nll_y(p) + nll_l(p(1:n)) + nll_s(p(n+1:2*n));

% p0 = [1e0*ones([n,1]); 1e0*ones([n, 1]); 0e0*ones([n, 1])];
p0 = [1e0*ones([n,1]); 1e0*ones([n, 1])];

p_opt = fmincon( nll, p0 );

%% NSGP with optimized hyperparameters

l = p_opt(1:n);
s = p_opt(n+1:2*n);
% o = p_opt(2*n+1:end);

lt = mean(l) + Kl_t' * (inv(Kl) *(l-mean(l)));
st = mean(s) + Ks_t' * (inv(Ks) *(s-mean(s)));
ot = mean(o) + Ko_t' * (inv(Ko) *(o-mean(o)));

K = kernel(x_train, x_train, l, l, s, s) + diag(o);
K_s = kernel(x_train, x, l, lt, s, st);
K_ss = kernel(x, x, lt, lt, st, st) + diag(ot);

K_inv = inv(K);
mu_s = K_s' * K_inv * y_train;
cov_s = K_ss - K_s' * K_inv * K_s;

lb = mu_s - 2*sqrt( diag(cov_s) );
ub = mu_s + 2*sqrt( diag(cov_s) );

figure();
hold on;
fill( [x; flipud(x)], [lb; flipud(ub)], 'y');
plot( x, mu_s, '-r', 'linewidth', 2);
plot( x, test_func((x+5)/10), '-m' );
plot( x_train, y_train, 'o');