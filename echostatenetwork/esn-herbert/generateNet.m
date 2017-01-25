% Create a ESN network. Call of this script returns: 
% 1. intWM0, sparse weight matrix of reservoir scaled to spectral radius 1, 
% 2. inWM random matrix of input-to-reservoir weights, 
% 3. ofbWM, output feedback weight matrix, 
% 4. initialOutWM all-zero initial weight matrix of reservoir-to-output units (is replaced
% by learnt weights as result of learning)

% netDim is number of neurons inside the reservoir.
netDim = 200; 
% connectivity is average connectivity inside reservoir. A standard choice
% is to set it such that on average a reservoir unit is connected to 10
% others
connectivity = min([10/netDim 1]);
% inputLenght and outputLenght are dimensions of input/output vectors
inputLength = 2;
outputLength = 2;
totalDim = netDim + inputLength + outputLength;

disp('Creating network ............');
disp(sprintf('netDim %g  Connectivity  %g', netDim, connectivity  ));


% the following creation of intWM0 is tried three times 
% because the call of myeigs sometimes (about every 20th time) 
% fails with randomly created matrices. minusPoint5 is a little helper
% function that subtracts 0.5 from entries of a sparse matrix.
% myeigs is actually essentially the same as the Matlab "eigs" function,
% which I modified a bit by suppressing output; "eigs" is sometimes very
% verbose.

try,
    intWM0 = sprand(netDim, netDim, connectivity);
    intWM0 = spfun(@minusPoint5,intWM0);
    maxval = max(abs(myeigs(intWM0,1)));
    intWM0 = intWM0/maxval;
catch,
    try,
        intWM0 = sprand(netDim, netDim, connectivity);
        intWM0 = spfun(@minusPoint5,intWM0);
        maxval = max(abs(myeigs(intWM0,1)));
        intWM0 = intWM0/maxval;
    catch,
        try,
            intWM0 = sprand(netDim, netDim, connectivity);
            intWM0 = spfun(@minusPoint5,intWM0);
            maxval = max(abs(myeigs(intWM0,1)));
            intWM0 = intWM0/maxval;
        catch,
        end
    end
end


% input weight matrix has weight vectors per input unit in colums
inWM = 2.0 * rand(netDim, inputLength)- 1.0;

% output weight matrix has weights for output units in rows
% includes weights for input-to-output connections
initialOutWM = zeros(outputLength, netDim + inputLength);

%output feedback weight matrix has weights in columns
ofbWM0 = (2.0 * rand(netDim, outputLength)- 1.0);
