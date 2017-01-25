function [neto, max_eig] = init(net, probRec, rngRec, probInp, rngInp, probBack, rngBack)
% ESN.INIT - initialze network weights and initial acivations

% initialize weights according to the input parameters
net.weightsRecurrent = init_weights(net.weightsRecurrent, probRec,  rngRec);
net.weightsInput     = init_weights(net.weightsInput,     probInp,  rngInp);
net.weightsBackward  = init_weights(net.weightsBackward,  probBack, rngBack);
net.actInital        = 2.0 * rand(net.numUnitsHidden, 1) - 1.0;

fid=fopen('weights.txt', 'w')
fprintf(fid, 'hiddenweights = ['); 
for i=1:100
    fprintf(fid, '[');    
    for j=1:100
        fprintf(fid, '%f,', net.weightsRecurrent(j,i));
    end
    fprintf(fid,'],');
end
fprintf(fid, ']\n'); 

fprintf(fid, 'backweights = ['); 
for i=1:100
    fprintf(fid, '%f,', net.weightsBackward(i));
end
fprintf(fid, ']\n'); 

fprintf(fid, 'inputweights = ['); 
for i=1:100
    fprintf(fid, '%f,', net.weightsInput(i));
end
fprintf(fid, ']\n'); 

fprintf(fid, 'actinitial = ['); 
for i=1:100
    fprintf(fid, '%f,', net.actInital(i));
end
fprintf(fid, ']\n'); 

fclose(fid);



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