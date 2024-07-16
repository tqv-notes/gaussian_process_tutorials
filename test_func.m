function x = test_func(x)
n = length(x);
i = find(x <= 0.6);
x(i) = 2*sin(pi*0.8*x(i)*4) + 0.4*cos(pi*0.8*x(i)*16);
x(setdiff(1:n,i)) = 2*x(setdiff(1:n,i)) - 1;
end