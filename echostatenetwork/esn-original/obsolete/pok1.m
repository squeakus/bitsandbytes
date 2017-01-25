clear classes;

TS = repmat((0.5 * sin(pi / 10 * (1:2*300)).^7),2,1);
IS = zeros(0, size(TS,2));

net = esn(0,100,2);
[net, me] = init(net, 0.05, -0.4, 0, 0, 1, -1);
disp(sprintf('Maximal eigenvalue is %f', me));
%net = scaleeig(net, 0.8);

[net, MSE] = train(net, IS , TS, 100);
[net, MSE, OS] = test (net, IS , TS, 100);

MSE

plot(OS(1,:),'r');
