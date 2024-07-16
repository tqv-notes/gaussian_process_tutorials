%% simple implementation of Multi-Output Gaussian Process (MOGP) regression

% Gaussian kernel
kernel = @(x1,x2,l,sigma_f) sigma_f^2 * exp( -0.5 * l^2 * ( (sum(x1.^2,2)) + (sum(x2.^2,2)).' - 2*x1*(x2.') ) );

kernel_params_true = [2 0.8]; % [sigma_f l]
n_features = 2;
n_latent_dims = 1;
n = 50;
ntest = 50;
n_spatial_dims = 1;
sigma2 = 0.01;

x_full = linspace(-7, 7, n+ntest)';
W_orig = [-2 2];
F_orig = mvnrnd(zeros([n+ntest 1]),kernel(x_full, x_full, kernel_params_true(2), kernel_params_true(1)),n_latent_dims).';

y_full = F_orig * W_orig + sqrt( sigma2 ) * randn([n+ntest 2]);

x = x_full(1:n);
y = y_full(1:n,:);
xtest = x_full(n+1:end);
ytest = y_full(n+1:end,:);

figure();
hold on;
plot( x, y );
plot( xtest, ytest );

%% optimize hyperparameters

cov_xx = @(p) kernel(x, x, p(2), p(1)) + p(3) * eye(size(x,1));
cov = @(p) kron( p(4:5).' * p(4:5), cov_xx(p(1:3)) ) + 1e-4 * eye(size(x,1) * n_features);

yn = y;
nll = @(p) 0.5 * log(det(cov(p))) + 0.5*yn(:).' * inv(cov(p)) * yn(:) + 0.5 * length(x) * n_features * log(2*pi);

p_init = [1 1 0.01 -1 1];
p_lb = [0   0   1e-4 -1e2 -1e2];
p_ub = [1e1 1e1 1e1   1e2  1e2];

p_opt = fmincon(nll, p_init, [], [], [], [], p_lb, p_ub);

W_fitted = p_opt(4:5);
noise_variance = p_opt(3);
kernel_params = p_opt(1:2);

%% make predictions

xnew = linspace(-10,10,300)';

WWT = W_fitted.' * W_fitted;

xaugmented = [x; xtest];
Kxx_augmented = kernel(xaugmented, xaugmented, kernel_params(2), kernel_params(1));
Kxx_augmented_full = kron( WWT, Kxx_augmented );

Kxx = Kxx_augmented_full(1:n*n_features+ntest,1:n*n_features+ntest) + 1e-4*eye(n*n_features+ntest);
Kxx = Kxx + noise_variance * eye(size(Kxx,1));

Kxx_eval = [WWT(1,2)*kernel([x; xtest], xnew, kernel_params(2), kernel_params(1));
            WWT(2,2)*kernel(x, xnew, kernel_params(2), kernel_params(1))];

Kxx_inv = inv(Kxx);

y_for_preds = [y(:,1); ytest(:,1); y(:,2)];
y2_mogp = Kxx_eval.' * Kxx_inv * y_for_preds;

% get predictions from standard GP
Kxx = kernel(x, x, kernel_params(2), kernel_params(1));
Kxx = Kxx + noise_variance * eye(size(Kxx,1));
Kxx_inv = inv(Kxx);
Kxxnew = kernel(x, xnew, kernel_params(2), kernel_params(1));
y1_gp = Kxxnew.' * Kxx_inv * y(:,1);
y2_gp = Kxxnew.' * Kxx_inv * y(:,2);

figure();
hold on;
plot([x; xtest], [y(:,1); ytest(:,1)], '-ob', 'markerfacecolor', 'b');
plot(x, y(:,2), '-or', 'markerfacecolor', 'r');
plot([x; xtest], [y(:,2); ytest(:,2)], '-or');
plot(xnew, y2_mogp, '-', 'linewidth', 2);
plot(xnew, y2_gp, '-', 'linewidth', 2);
xlabel('x');
ylabel('y');
legend({'y_1 (train)', 'y_2 (train)', 'y_2 (test)', 'y_2 predicted with MOGP', 'y_2 predicted with GP'}, 'location', 'best');
box on;
set(gcf,'color','w');

