% training ESN on sin^7 time series

% creating input and output time series
TS = 0.5  * sin(pi/10*(1:300)).^7 + 0.2*rand(1,300);
IS = zeros(0, size(TS,2));

% create ESN object, 0 input, 100 hidden and 1 output unit
net = esn(0,100,1);

% initialize ESN weights
% 5% of recurrent weights set to 0.4 or -0.4
% input weights set to 0
% 100% backward weights set to 1 or -1
% no treshold, print maximal recurrent weights eigenvalue 
[net, me] = init(net, 0.05, -0.4, 0, 0, 1, -1);
net = settreshold(net, 0);
fprintf('Maximal eigenvalue is %f.\n', me);

% train ESN on given time series, supress 100 initial steps from training
[net, MSE] = train(net, IS , TS, 100);

% test ESN on the same time series, supress 100 initial steps from MSE
% calculation
[net, MSE, OS] = test (net, IS , TS, 100);

% print MSE and plot output signal
fprintf('Testing MSE is %.20f.\n', MSE);
plot(TS(1,:),'b--'); hold on;
plot(OS(1,:),'r');
