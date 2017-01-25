function [neto, max_eig] = init(net, probRec, rngRec, probInp, rngInp, probBack, rngBack)
% ESN.INIT - initialze network weights and initial acivations

% initialize weights according to the input parameters
net.weightsRecurrent = init_weights(net.weightsRecurrent, probRec,  rngRec);
net.weightsInput     = init_weights(net.weightsInput,     probInp,  rngInp);
net.weightsBackward  = init_weights(net.weightsBackward,  probBack, rngBack);

% initialize initial activations from [-1, 1]
net.actInital        = 2.0 * rand(net.numUnitsHidden, 1) - 1.0;

% get maximal eigenvalue 
max_eig = max(abs(eig(net.weightsRecurrent)));
neto = net;

function weights = init_weights(weights, prob, rng)
% initialze weights given as inputs

mask = rand(size(weights)) < prob;
weights = (2.0 * rand(size(weights)) - 1.0) .* mask;
if rng >= 0, 
    weights = weights .* rng;
else 
    weights(weights < 0) = rng; 
    weights(weights > 0) = -rng;
end;
