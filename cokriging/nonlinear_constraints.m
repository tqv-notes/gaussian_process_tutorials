function [c, ceq] = nonlinear_constraints(x, variofun_list)
% c(x) <= 0
% ceq(x = 0

nr_constraints = length(variofun_list);
c = zeros([nr_constraints 1]);

start_idx = 0;
for idx = 1:nr_constraints
    c(idx) = x(start_idx + 3)^2 - x(start_idx + 1) * x(start_idx + 2);
    start_idx = start_idx + 3 + variofun_list{idx}{2};
end

ceq = [];

end