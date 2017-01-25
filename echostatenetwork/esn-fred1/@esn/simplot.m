function [neto, OS, AH] = simplot(net, IS, TS, dsteps, alpha, rows, cols)
% ESN.SIMPLOT - simulating = working phase of trained ESN
% IS  - input sequence, can be empty matrix
% TS  - target sequence for starting period
% dsteps - number of dummy steps
% rows - no of rows of figures
% cols = no of cols of figures  (we start from first hidden unit)
% OS  - output sequence
% AH - hidden units activitiy


% check object
if ~ isa(net, 'esn'),
    error('Not an esn object.');
end;

% check sizes of input and output sequence
if size(IS,2) < size(TS,2),
    error('Input sequence must be longer or equal to the output sequence.');
end;

% check dimension of input sequence
if size(IS,1) ~= net.numUnitsInput,
    error('Input sequence dimension and number of input units do not match.');
end;

% check dimension of output sequence
if size(TS,1) ~= net.numUnitsOutput,
    error('Target sequence dimension and number of output units do not match.');
end;

% check length of input
if size(IS,2) < 2,
    perror('Input sequence is too short.');
end;

% check dsteps
if dsteps >= size(IS,2) || dsteps > size(TS,2),
    error('Number of dummy steps is too high.');
end;

% set alpha if not given
if nargin < 4, 
    alpha = 1; 
end;

% add ones as bias inputs
IS = [IS; ones(1, size(IS,2))];

% find number of steps
numStepsI = size(IS,2);
numStepsT = dsteps;

% preallocate hidden units' activity matrix,
% fill first column with initial activations
AH = zeros(net.numUnitsHidden, numStepsI);
OS = zeros(net.numUnitsOutput, numStepsI);
AH(:,1) = net.actInital;
OS(:,1)  = TS(:,1);

% compute hidden units' activity matrix over time
for I = 2:numStepsT,
    AH(:,I) = (1 - alpha) * AH(:,I-1) + alpha * tanh(...
        net.weightsRecurrent * AH(:,I-1) + ...
        net.weightsInput * IS(:,I) + ...
        net.weightsBackward * TS(:,I-1) );
    OS(:,I) = tanh(net.weightsForward * [AH(:,I);IS(:,I)]);
end;

for I = numStepsT+1:numStepsI,
    AH(:,I) = (1 - alpha) * AH(:,I-1) + alpha * tanh(...
        net.weightsRecurrent * AH(:,I-1) + ...
        net.weightsInput * IS(:,I) + ...
        net.weightsBackward * OS(:,I-1) );
    OS(:,I) = tanh(net.weightsForward * [AH(:,I);IS(:,I)]);
end;

count = 0;
for r = 1:rows,
    for c = 1:cols,
        count = count + 1;
        subplot(rows,cols,count);
        plot(AH(count,1:40));
    end;
end;

% create output ESN object
neto = net;


