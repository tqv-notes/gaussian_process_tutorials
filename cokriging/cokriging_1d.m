%% generate data

rng(12345,'twister');

% note: (X1,Y1) is the secondary data and (X2,Y2) is the primary data

X1 = (0:0.01:1.0)';
X2 = (0:0.02:0.6)';

Y1 = cos(2*pi*X1 + 1.0) + randn(size(X1)) * 0.03;
Y2 = cos(2*pi*X2 + 1.5) + randn(size(X2)) * 0.03;

figure();
hold on;
plot(X1, Y1, '-*');
plot(X2, Y2, '-s');
xlabel('x');
ylabel('y');
legend('secondary data', 'primary data');
set(gcf,'color','w');
box on;

%% variogram and covariance

vario_matern = @(c,h) 1 - 2^(1-c(1))/gamma(c(1)) * ( sqrt(2*c(1)) * (h+eps)/c(2) ).^c(1) .* besselk( c(1), sqrt(2*c(1)) * (h+eps)/c(2) );

cov_matern = @(c,h) 2^(1-c(1))/gamma(c(1)) * ( sqrt(2*c(1)) * (h+eps)/c(2) ).^c(1) .* besselk( c(1), sqrt(2*c(1)) * (h+eps)/c(2) );

%% estimate variogram and cross-variogram

% variogram for primary and secondary data
for data_idx = 1:2

    if data_idx == 1
        X = X1;
        Y = Y1;
    elseif data_idx == 2
        X = X2;
        Y = Y2;
    end

    ns = length(X);

    dist = zeros([ns ns]);
    for idx = 1:ns
        dist(idx,:) = abs( X(idx,1) - X(:,1) );
    end

    vario_gamma = zeros([ns ns]);
    for idx = 1:ns
        vario_gamma(idx,:) = ( Y(idx,1) - Y(:,1) ) .* ( Y(idx,1) - Y(:,1) );
    end

    dist = dist(:);
    vario_gamma = vario_gamma(:);

    bins = linspace(0, max(dist)/2, 21); % factor 1/2 is to limit variogram estimation to cases with sufficiently large number of samples

    x = zeros([length(bins)-1 1]);
    y = zeros(size(x));
    e = zeros(size(x));
    for idx = 1:length(bins)-1
        select = dist >= bins(idx) & dist < bins(idx+1);
        x(idx) = 0.5*(bins(idx) + bins(idx+1));
        y(idx) = 0.5*mean( vario_gamma(select) );
        e(idx) = std( vario_gamma(select) )/sqrt( length(vario_gamma(select)) );
    end

    if data_idx == 1
        x_11 = x;
        y_11 = y;
        e_11 = e;
    elseif data_idx == 2
        x_22 = x;
        y_22 = y;
        e_22 = e;
    end

end

% cross-variogram

% get common measured positions
dist = abs(X1 - X2');
select_1 = (prod(dist, 2) == 0);
select_2 = (prod(dist, 1)' == 0);

X1_new = X1(select_1);
Y1_new = Y1(select_1);
X2_new = X2(select_2);
Y2_new = Y2(select_2);

assert( all( abs(X1_new - X2_new)<1e-16 ) ); % sanity check

ns = length(X2_new); % should be the same for X1_new and X2_new

dist = zeros([ns ns]);
for idx = 1:ns
    dist(idx,:) = abs( X2_new(idx,1) - X2_new(:,1) );
end

vario_gamma = zeros([ns ns]);
for idx = 1:ns
    vario_gamma(idx,:) = ( Y1_new(idx,1) - Y1_new(:,1) ) .* ( Y2_new(idx,1) - Y2_new(:,1) );
end

dist = dist(:);
vario_gamma = vario_gamma(:);

bins = linspace(0, max(dist)/2, 21); % factor 1/2 is to limit variogram estimation to cases with sufficiently large number of samples

x_12 = zeros([length(bins)-1 1]);
y_12 = zeros(size(x_12));
e_12 = zeros(size(x_12));

for idx = 1:length(bins)-1
    select = dist >= bins(idx) & dist < bins(idx+1);
    x_12(idx) = 0.5*(bins(idx) + bins(idx+1));
    y_12(idx) = 0.5*mean( vario_gamma(select) );
    e_12(idx) = std( vario_gamma(select) )/sqrt( length( vario_gamma(select) ) );
end

% remove NaN
nanlist = isnan(y_12);
x_12(nanlist) = [];
y_12(nanlist) = [];
e_12(nanlist) = [];

nanlist = isnan(y_11);
x_11(nanlist) = [];
y_11(nanlist) = [];
e_11(nanlist) = [];

nanlist = isnan(y_22);
x_22(nanlist) = [];
y_22(nanlist) = [];
e_22(nanlist) = [];

figure();
hold on;
errorbar( x_11, y_11, e_11, '-o', 'markerfacecolor', 'b' );
errorbar( x_22, y_22, e_22, '-o', 'markerfacecolor', 'r' );
errorbar( x_12, y_12, e_12, '-o', 'markerfacecolor', 'y' );
legend({'variogram for secondary data', 'variogram for primary data', 'cross-variogram'}, 'location', 'best');
legend('boxoff');
box on;
set(gcf,'color','w');
xlabel('x');
ylabel('variogram');
title('variogram and cross-variogram');

%% fit variogram and cross-variogram

variofun_list = { {vario_matern, 2}, {vario_matern, 2} };

objectfun = @(b) variogram_cost_func( x_11, y_11, x_22, y_22, x_12, y_12, variofun_list, b );

MAX_ITER = 10; % number of optimization repeats with random initialization

cost = NaN([MAX_ITER 1]);
params_opt = [];

for iter = 1:MAX_ITER

    b0 = [];
    ub = [];
    for idx = 1:length(variofun_list)
        b0 = [b0 max(y_11)*rand() max(y_22)*rand() max(y_12)*rand() 10*rand() max(x_22)*rand()];
        ub = [ub max(y_11) max(y_22) max(y_12) 10 max(x_22)]; %#ok<*AGROW> 
    end
    lb = zeros(size(ub));

    nonlcon = @(x) nonlinear_constraints(x, variofun_list); % to enforce the positive definiteness of LCM matrices

    fmincon_opts = optimoptions('fmincon', 'Display', 'off', 'MaxFunctionEvaluations', 10000);

    b = fmincon( objectfun, b0, [], [], [], [], lb, ub, nonlcon, fmincon_opts );

    cost(iter) = objectfun(b);
    params_opt{iter} = b; %#ok<*SAGROW> 

end

% get the best result
[~, ind] = min(cost);
b = params_opt{ind};

% check fitting
[gamma_11, gamma_22, gamma_12] = get_gamma_models( x_11, x_22, x_12, variofun_list, b );

figure();
set(gcf,'color','w');

subplot(1,3,1);
hold on;
plot(x_11, y_11, 'o');
plot(x_11, gamma_11);
legend({'data','fit'},'location','best');
legend('boxoff');
box on;
title('gamma_{11} (secondary)');
xlabel('x');
ylabel('gamma');

subplot(1,3,2);
hold on;
plot(x_22, y_22, 'o');
plot(x_22, gamma_22);
legend({'data','fit'},'location','best');
legend('boxoff');
box on;
title('gamma_{22} (first)');
xlabel('x');
ylabel('gamma');

subplot(1,3,3);
hold on;
plot(x_12, y_12, 'o');
plot(x_12, gamma_12);
legend({'data','fit'},'location','best');
legend('boxoff');
box on;
title('gamma_{12} (cross-term)');
xlabel('x');
ylabel('gamma');

%% ordinary kriging

% perform ordinary kriging for the primary data with the theorical variogram models obtained in the previous step

X = X2;
Y = Y2;

ns = length(X);

Dx = abs( bsxfun( @minus, X, X' ) );

func = @(h) get_model( h, variofun_list, b, 2); % gamma_22

% variogram matrix
A = func(Dx);

% noise/nugget term (note: this term can be estimated in the optimization
% step but for simplicity, we guess this term based on visual inspection)
sigma_noise = 5e-3;
noise = sigma_noise * ( 1 - eye(ns) );
A = A + noise;

% the matrix must be expanded by one line and one row to account for the
% condition that sum of weights is equal to 1 (Lagrange multiplier)
A = [[A ones(ns,1)]; ones(1, ns) 0];

% we also need to expand output
Yn = [Y; 0];

% test datapoints
Xt = linspace(min(X1), max(X1), 100).';
dt = abs( bsxfun( @minus, X, Xt' ) );

d = func(dt) + sigma_noise;
d = [d; ones(1, length(Xt))];

% solve system
lambda = pinv(A) * d;

% prediction
Yt = lambda' * Yn;

% sigma = sqrt( sum( d .* lambda, 1 ) )';

% for later plot
X_kriging = Xt;
Y_kriging = Yt;

% figure();
% hold on;
% plot(X, Y, 'o');
% plot(Xt, Yt, '-s');
% box on;
% set(gcf,'color','w');
% xlabel('x');
% ylabel('y');
% legend('primary data','prediction');
% title('kriging prediction');

%% ordinary cokriging

n1 = length(X1);
n2 = length(X2);
ns = n1 + n2;

func_11 = @(h) get_model(h, variofun_list, b, 1);
func_22 = @(h) get_model(h, variofun_list, b, 2);
func_12 = @(h) get_model(h, variofun_list, b, 3);

Dx = abs( bsxfun( @minus, X2, X2' ) );
C22 = func_22(Dx);

Dx = abs( bsxfun( @minus, X1, X1' ) );
C11 = func_11(Dx);

Dx = abs( bsxfun( @minus, X2, X1' ) );
C21 = func_12(Dx);

% variogram matrix
A = [C22  C21;
     C21' C11];

sigma_noise = 2e-2;
noise = sigma_noise * ( 1 - eye(ns) );
A = A + noise;

% the matrix must be expanded by one line and one row to account for the
% condition that sum of weights is equal to 1 (Lagrange multiplier)
a1 = [ones(n2, 1); zeros(n1, 1)];
a2 = [zeros(n2, 1); ones(n1, 1)];
A = [ A [a1 a2]; [a1 a2]' zeros(2,2)];

Yn = [Y2; Y1; 0 ; 0];

% test datapoints
Xt = linspace(min(X1), max(X1), 100).';

d22 = abs( bsxfun( @minus, X2, Xt' ) );
d22 = func_22(d22) + sigma_noise;

d21 = abs( bsxfun( @minus, X1, Xt' ) );
d21 = func_12(d21);

d = [d22; d21];

d = [d; ones(1, length(Xt)); zeros(1, length(Xt))];

% solve system
lambda = pinv(A) * d;

% prediction
Yt = lambda' * Yn;

sigma = sqrt( sum( d .* lambda, 1) );

figure();
hold on;
box on;
set(gcf,'color','w');
plot(X2, Y2, 'o');
plot(X1, Y1, '*');
plot(Xt, Yt, '-k', 'linewidth', 1.5);
plot(X_kriging, Y_kriging, '-m', 'linewidth', 1.5);
xlabel('x data');
ylabel('y data');
legend({'primary data', 'secondary data','cokriging prediction', 'kriging prediction'},'location','best');
legend('boxoff');

