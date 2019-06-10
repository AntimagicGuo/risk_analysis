function y = zhongxinju(x,k)
%k:k jie
%x：input
n = length(x);
x_mean = sum(x)/n;
y = sum((x-x_mean).^k)/n;
disp('x的k阶中心矩为：'),disp(y)
end
