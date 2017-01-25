clear classes;

S = loadseq('mel.trs','BIN');
S = repmat(S,1,100);
S = tanh(S .* 2 - 1);
S = S + rand(size(S)) .* 0.002 - 0.001;

IS = zeros(0,1500);
TS = S(:,1:1500);

net = esn(0,400,1);
[net, me] = init(net, 0.0125, -0.4, 0, 0, 1, 4);
disp(sprintf('Maximal eigenvalue is %f', me));

[net, MSE] = train(net, IS , TS, 500);
[net, MSE, OS] = test (net, IS , TS, 500);

MSE

plot(OS(1,:),'r'); hold on;
plot(OS(2,:),'g');