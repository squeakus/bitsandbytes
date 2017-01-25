function [neto, MSE, OS, AH] = train(net, IS, TS, dsteps, noise, alpha, noforwardfrominput)
% ESN.TRAIN - ESN training
% IS  - input sequence, can be empty matrix
% TS  - target sequence
% dsteps - number of dummy steps
% noise - noise added to hidden unit activities, regular dist.
% alpha - hidden unit activities change ratio, 1.0=total change
% noforwardfrominput - do not train weights from input to output units

% MSE - how well net is trained
% OS - output sequence (activities are calculated)
% AH - recurrent units activities


% check object
if ~ isa(net, 'esn'),
    error('Not an esn object.');
end;

% check sizes of input and output sequence
if size(IS,2) ~= size(TS,2),
    error('Input and output sequences must have the same length.');
end;

% check dimension of input sequence
if size(IS,1) ~= net.numUnitsInput,
    error('Input sequence dimension and number of input units do not match.');
end;

% check dimension of output sequence
if size(TS,1) ~= net.numUnitsOutput,
    error('Target sequence dimension and number of output units do not match.');
end;

% check length of input (and output sequence)
if size(IS,2) < 2,
    error('Input and output sequences are too short.');
end;

% check dsteps
if dsteps >= size(IS,2),
    error('Number of dummy steps is too high.');
end;

% set noise if not given
if nargin < 5, 
    noise = 0; 
end;

% set alpha if not given
if nargin < 6, 
    alpha = 1; 
end;

% set noforwardfrominput if not given
if nargin < 7, 
    noforwardfrominput = 0;
end;

% add ones as bias inputs
IS = [IS; ones(1, size(IS,2))];

% find number of steps
numSteps = size(IS,2);

% preallocate hidden units' activity matrix,
% fill first column with initial activations
AH = zeros(net.numUnitsHidden, numSteps);
AH(:,1) = net.actInital;

% compute hidden units' activity matrix over time
for I = 2:numSteps,
    AH(:,I) = (1 - alpha) * AH(:,I-1) + alpha * tanh(...
        net.weightsRecurrent * AH(:,I-1) + ...
        net.weightsInput * IS(:,I) + ...
        net.weightsBackward * TS(:,I-1) + ...
        (2*rand-1) * noise  );
end;

% throw out dummy activities
IS = IS(:,dsteps+1:end);
AH = AH(:,dsteps+1:end);
TS = TS(:,dsteps+1:end);

% prepare arry of raw activities 
PTS = atanh(TS);

% fit forward weights 
if noforwardfrominput == 0,
    net.weightsForward = ([AH', IS'] \ PTS')';
else
    net.weightsForward(:,1:net.numUnitsHidden) = (AH' \ PTS')';
end;

% compute output activities by using forward weights
% and calculate mean square error
OS = zeros(size(TS));
OS(:,1) = TS(:,1);
AH(:,2:end) = 0;

for I = 2:size(IS,2),
    AH(:,I) = (1 - alpha) * AH(:,I-1) + alpha * tanh(...
        net.weightsRecurrent * AH(:,I-1) + ...
        net.weightsInput * IS(:,I) + ...
        net.weightsBackward * TS(:,I-1) + ...
        (2*rand-1) * noise  );
end;

OS = tanh(net.weightsForward * [AH; IS]);
MSE = sum(sum((TS-OS).^2)) ./ prod(size(OS));

% create output ESN object
neto = net;


