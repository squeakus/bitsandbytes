
[X, Y, Z] = create_lorenz(10000, 100);
X = X ./ 50;
Y = Y ./ 50;
Z = Z ./ 50;
IS = [Y; Z];
TS = [X];

[net, MSETST, OS] = test(net, IS , TS, 2000, 1.0);
fprintf('Testing error os %.10f\n', MSETST);

subplot(4,1,1);   plot(OS(1,1:end));
subplot(4,1,2);   plot(X(1:end));
subplot(4,1,3);   plot(Y(1:end));
subplot(4,1,4);   plot(Z(1:end));
