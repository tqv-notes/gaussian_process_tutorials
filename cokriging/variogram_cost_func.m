function cost = variogram_cost_func( x_11, y_11, x_22, y_22, x_12, y_12, func_list, params )

% build theoretical variogram models
[gamma_11, gamma_22, gamma_12] = get_gamma_models(x_11, x_22, x_12, func_list, params);

cost = mean( (y_11 - gamma_11).^2 ) + ...
       mean( (y_22 - gamma_22).^2 ) + ...
       mean( (y_12 - gamma_12).^2 );

end