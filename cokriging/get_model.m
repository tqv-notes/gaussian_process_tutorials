function gamma = get_model(h, func_list, params, case_idx)

gamma = zeros(size(h));

for idx = 1:length(func_list)

    vario_func = func_list{idx}{1};
    nr_hypers = func_list{idx}{2};
    c = params(1:3);
    b = params(4:3+nr_hypers);

    gamma = gamma + c(case_idx)*vario_func(b,h);

    params = params(3+nr_hypers+1:end);

end

end