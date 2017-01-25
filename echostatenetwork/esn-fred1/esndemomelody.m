% training ESN on 'melody' attractor 

% load melody (file is character array), add some noise
MEL_S = loadseq('mel.trs', 'REAL');
MEL_S = repmat(MEL_S,1,100);
MEL_S = tanh(MEL_S .* 2 - 1);
MEL_S = MEL_S + rand(size(MEL_S)) .* 0.002 - 0.001;

% check melody sequence length
if size(MEL_S, 2) < 1500,
    error('Melody sequence is too short.');
end;

% ctrate input and output sequence
IS = zeros(0,1500);
TS = MEL_S(:,1:1500);

% create ESN object, 0 input, 400 hidden and 1 output unit, no treshold 
net = esn(0,400,1);

% initialize ESN weights
% 1.25% of recurrent weights set to 0.4 or -0.4  [FC: one or the
% other only]
% input weights set to 0
% 100% backward weights set to 4 or -4  [FC: in range -4..4]
% no treshold, print maximal recurrent weights eigenvalue 
[net, me] = init(net, 0.0125, -0.4, 0, 0, 1, 4);
net = settreshold(net, 0);
fprintf('Maximal eigenvalue is %f.\n', me);

% train ESN on given time series, supress 500 initial steps from training
[net, MSE] = train(net, IS , TS, 500);

% test ESN on the same time series, supress 500 initial steps from MSE
% calculation
[net, MSE, OS] = test (net, IS , TS, 500);

% print MSE and plot output signal
fprintf('Testing MSE is %.20f\n', MSE);
plot(TS(1,:),'b:'); hold on;
plot(OS(1,:),'r');
