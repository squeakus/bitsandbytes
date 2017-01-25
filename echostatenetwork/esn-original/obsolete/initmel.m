function neto = initmel(net)
%
%

maskWeightsRecurrent = rand(net.numUnitsHidden, net.numUnitsHidden) < 0.0125;
net.weightsRecurrent = rand(net.numUnitsHidden, net.numUnitsHidden);
net.weightsRecurrent(net.weightsRecurrent > 0.5) = 1;
net.weightsRecurrent(net.weightsRecurrent <= 0.5) = -1;
net.weightsRecurrent = net.weightsRecurrent .* 0.4 .* maskWeightsRecurrent;

max_eigv = max(abs(eig(net.weightsRecurrent)));
disp(sprintf('Maximal eigenvalue is %f', max_eigv));

maskWeightsInput     = rand(net.numUnitsHidden, net.numUnitsInput+1);
net.weightsInput     = (2.0 * rand(net.numUnitsHidden, net.numUnitsInput+1) - 1.0) .* maskWeightsInput;

maskWeightsBackward  = rand(net.numUnitsHidden, net.numUnitsOutput);
net.weightsBackward  = (4.0 * rand(net.numUnitsHidden, net.numUnitsOutput) - 2.0) .* maskWeightsBackward;

net.actInital        = 2.0 * rand(net.numUnitsHidden, 1) - 1.0;

neto = net;