function [gamma_11, gamma_22, gamma_12] = get_gamma_models(x_11, x_22, x_12, func_list, params)

gamma_11 = zeros(size(x_11));
gamma_22 = zeros(size(x_22));
gamma_12 = zeros(size(x_12));

for idx = 1:length(func_list)

    vario_func = func_list{idx}{1};
    nr_hypers = func_list{idx}{2};
    c = params(1:3);
    b = params(4:3+nr_hypers);

    gamma_11 = gamma_11 + c(1)*vario_func(b,x_11);
    gamma_22 = gamma_22 + c(2)*vario_func(b,x_22);
    gamma_12 = gamma_12 + c(3)*vario_func(b,x_12);

    params = params(3+nr_hypers+1:end);

end

end