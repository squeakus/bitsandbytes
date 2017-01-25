clear classes;

IS = zeros(0,6000);
TS = atan(createmg(6000, 10, 0.2, 10, 0.1, 30, 100)-1);
TS = TS + (2*rand(size(TS))-1) .* 0.0001;

net = esn(0,400,1);
[net, me] = init(net, 0.0125, -0.4, 0.25, -0.028, 1, 0.56);
disp(sprintf('Maximal eigenvalue is %f', me));
net = scaleeig(net, 0.79);

[net, MSETRN] = train(net, IS , TS, 1000, 0.00, 0.4);
disp(sprintf('Training error is %f', MSETRN));



IS = zeros(0,6000);
TS = atan(createmg(6000, 10, 0.2, 10, 0.1, 30, 100)-1);

[net, MSETST, OS] = test(net, IS , TS, 1000, 0.4);
disp(sprintf('Testing error os %f', MSETST));

subplot(2,1,1); plot(OS(1000:end));
subplot(2,1,2); plot(TS(1000:end));

