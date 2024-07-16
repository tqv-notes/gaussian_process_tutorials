%% simple implementation of Gaussian Process (GP) regression

% Gaussian kernel
kernel = @(x1,x2,l,sigma_f) sigma_f^2 * exp( -0.5 * l^2 * ( (sum(x1.^2,2)) + (sum(x2.^2,2)).' - 2*x1*(x2.') ) );

x = (-5:0.2:5)';
mu = zeros(size(x));
cov = kernel(x,x,1,1);

samples = mvnrnd(mu,cov,3);

figure();
plot(samples');
title('samples from a gaussian process');
axis('tight');

%% GP with default hyperparameters

x_train = [-4 -3 -2 -1 1]';
y_train = sin(x_train);

sigma_y = 1e-3;
K = kernel(x_train, x_train, 1, 1) + sigma_y * eye(length(x_train));
K_s = kernel(x_train, x, 1, 1);
K_ss = kernel(x, x, 1, 1) + sigma_y * eye(length(x));

K_inv = inv(K);
mu_s = K_s' * K_inv * y_train;
cov_s = K_ss - K_s' * K_inv * K_s;

lb = mu_s - 2*sqrt( diag(cov_s) );
ub = mu_s + 2*sqrt( diag(cov_s) );

figure();
hold on;
fill( [x; flipud(x)], [lb; flipud(ub)], 'y' );
plot( x, mu_s, '-r', 'linewidth', 2 );
plot( x, sin(x), '-om' );

%% hyperparameter optimization via negative log likelihood

Ker = @(params) kernel( x_train, x_train, params(1), params(2)) + sigma_y * eye(length(x_train));
nll = @(params) 0.5 * log(det(Ker(params))) + 0.5*y_train.' * inv(Ker(params)) * y_train + 0.5*length(x_train) * log(2*pi);

params_opt = fmincon( nll, [1 1] );

%% GP with optimized hyperparameters

K = kernel(x_train, x_train, params_opt(1), params_opt(2)) + sigma_y * eye(length(x_train));
K_s = kernel(x_train, x, params_opt(1), params_opt(2));
K_ss = kernel(x, x, params_opt(1), params_opt(2)) + sigma_y * eye(length(x));

K_inv = inv(K);
mu_s = K_s' * K_inv * y_train;
cov_s = K_ss - K_s' * K_inv * K_s;

lb = mu_s - 2*sqrt( diag(cov_s) );
ub = mu_s + 2*sqrt( diag(cov_s) );

figure();
hold on;
fill( [x; flipud(x)], [lb; flipud(ub)], 'y' );
plot( x, mu_s, '-r', 'linewidth', 2 );
plot( x, sin(x), '-om' );

