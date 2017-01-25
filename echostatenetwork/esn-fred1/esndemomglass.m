
% training ESN on Mackey Glass time series
rand('seed',13);

% creating input and output time series, add noise
TAU = 7; % mild
%TAU = 30; % wild
IS = zeros(0,4000);
TS = atan(createmg(4000, 10, 0.4, 10, 0.1, TAU, 100)-1);
TS = TS + (2*rand(size(TS))-1) .* 0.0001;
fileID = fopen('mgmatlab.dat','w');
fprintf(fileID,'%f\n',TS);
fclose(fileID);

% create ESN object, 0 input, 400 hidden and 1 output unit
%net = esn(0,400,1);
%net = scaleeig(net, 0.79);
net = esn(0,10,1);


% initialize ESN weights
% 1.25% of recurrent weights set to 0.4 or -0.4
% input weights set to 0
% 100% backward weights set randomly from [-0.56, 0.56]
% no treshold
[net, me] = init(net, 0.0125, -0.4, 0.0, 0.0, 1, 0.56);
fprintf('Maximal eigenvalue is %f\n', me);
net = settreshold(net, 0);

% scale recurrent weights eigenvalue and print old one
fprintf('Maximal eigenvalue before scaling to 0.79 is %f.\n', me);
net = scaleeig(net, 0.79);

disp(net);

% train ESN on given time series, 
% supress 1000 initial steps from training
% no noise to states 
% activations change ration set to 0.4
[net, MSETRN] = train(net, IS , TS, 1000, 0.00, 0.4);
fprintf('Training MSE is %.20f.\n', MSETRN);

% test ESN on the same time series
% supress 1000 initial steps from MSE calculation
% activations change ration set to 0.4 
[net, MSETST, OS] = test(net, IS , TS, 1000, 0.4);

% print MSE and plot output signal
fprintf('Testing MSE is %.20f.\n', MSETST);
plot(TS(1,:),'b:'); hold on;
plot(OS(1,:),'r');
