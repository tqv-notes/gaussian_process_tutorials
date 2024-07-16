%% generate data

X = (0:0.01:1)';
Y = cos(4*X -3) + randn(size(X)) * 0.1;

figure();
plot(X, Y, '-*');

%% estimate variogram

ns = length(X);

dist = zeros([ns ns]);
for idx = 1:ns
    dist(idx,:) = abs(X(idx,1) - X(:,1));
end

gamma = zeros([ns ns]);
for idx = 1:ns
    gamma(idx,:) = (Y(idx) - Y) .* (Y(idx) - Y);
end

dist = dist(:);
gamma = gamma(:);

bins = linspace(0, max(dist)/2,21); % factor 1/2 is used to limit to area with large number of samples

x = zeros([length(bins)-1 1]);
y = zeros(size(x));
e = zeros(size(x));

for idx = 1:length(bins)-1
    select = dist >= bins(idx) & dist < bins(idx+1);
    x(idx) = 0.5*(bins(idx) + bins(idx+1));
    y(idx) = 0.5*mean( gamma(select) );
    e(idx) = std(gamma(select))/sqrt(length(gamma(select)));
end

figure();
errorbar(x,y,e);

%% fit variogram

stablealpha = 2;
func_exp = @(b,h) b(2)*(1-exp(-(h.^stablealpha)/(b(1)^stablealpha)));

func_nugget = @(b,h) b*(abs(h)>0);

variofun = @(b,h) func_exp(b(1:2),h) + func_nugget(b(3),h);

b0 = [max(x)*2/3 max(y) 1e-3]; % initial guess
lb = zeros(size(b0)); lb(3) = 4e-3; % lower bound
ub = [inf max(y) max(y)];

options = optimoptions('fmincon','Display','off','MaxFunctionEvaluations',10000);
objectfun = @(b) sum( (variofun(b,x)-y).^2 );
b = fmincon(objectfun, b0, [], [], [], [], lb, ub, [], options);

range = b(1);
sill = b(2);
nugget = b(3);

% check fitting
figure()
hold on;
box on;
plot(x,y,'o');
plot(x,variofun(b,x));
legend({'data','fit'},'location','best');
legend('boxoff');
set(gcf,'color','w');

%% ordinary kriging with variogram

Dx = abs(bsxfun(@minus,X,X'));

% calculate matrix with variogram values
A = func_exp([range sill],Dx);
A = A + func_nugget(nugget,Dx);

% matrix must be expanded by one line and one row to account for condition
% that all weights must sum to 1 (lagrange multiplier)
A = [[A ones(ns,1)]; ones([1 ns]) 0];

% expand output for later convenience
Yn = [Y; 0];

% test data
Xt = linspace(min(X),max(X),100)';
Xt = sort([Xt; X]);
dt = abs(bsxfun(@minus,X,Xt'));

a = func_exp([range sill],dt) + nugget;
a = [a; ones(1, length(Xt))];

% solve system
lambda = pinv(A)*a;

% predict
Yt = lambda'*Yn;
sigma = sqrt( sum(a.*lambda,1) ).';

figure();
hold on;
box on;
plot(X,Y,'o');
plot(Xt,Yt,'-r','linewidth',1.5);
plot(Xt,Yt-3*sigma,'--m','linewidth',1);
plot(Xt,Yt+3*sigma,'--c','linewidth',1);
xlabel('X');
ylabel('Y');
legend({'data','y','y-3\sigma','y+3\sigma'},'location','best');
legend('boxoff');
set(gcf,'color','w');

%% ordinary kriging with covariogram

cov_exp = @(b,h) b(2)*exp(-(h.^stablealpha)/(b(1)^stablealpha));
cov_nugget = @(b,h) b*(abs(h)==0);

Dx = abs(bsxfun(@minus,X,X'));

% calculate matrix with variogram values
K = cov_exp([range sill],Dx);
K = K + cov_nugget(nugget,Dx);

K_inv = pinv(K);

% matrix must be expanded by one line and one row to account for condition
% that all weights must sum to 1 (lagrange multiplier)
K = [[K ones(ns,1)]; ones([1 ns]) 0];

% expand output for later convenience
Yn = [Y; 0];

% test data
Xt = linspace(min(X),max(X),100)';
Xt = sort([Xt; X]);
dt = abs(bsxfun(@minus,X,Xt'));

k = cov_exp([range sill],dt);
k = [k; ones(1, length(Xt))];

% solve system
lambda = pinv(K)*k;

% predict
Yt = lambda'*Yn;

dt = abs(bsxfun(@minus,Xt,Xt'));
kt = cov_exp([range sill],dt) + cov_nugget(nugget,dt);
k = k(1:end-1,:);
cov = kt - k' * K_inv * k;
sigma = sqrt( diag(cov) );

figure();
hold on;
box on;
plot(X,Y,'o');
plot(Xt,Yt,'-r','linewidth',1.5);
plot(Xt,Yt-3*sigma,'--m','linewidth',1);
plot(Xt,Yt+3*sigma,'--c','linewidth',1);
xlabel('X');
ylabel('Y');
legend({'data','y','y-3\sigma','y+3\sigma'},'location','best');
legend('boxoff');
set(gcf,'color','w');