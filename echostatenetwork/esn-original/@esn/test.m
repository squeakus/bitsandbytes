function [neto, MSE, OS, AH] = test(net, IS, TS, dsteps, alpha)
% ESN.TEST - ESN testing
% IS  - input sequence, can be empty matrix
% TS  - target sequence
% dsteps - number of dummy steps
% MSE - how well net is trained
% OS  - output sequence
% AH - hidden units activities

% check object
if ~ isa(net, 'esn'),
    error('Not an esn object.');
end;

% set alpha if not given
if nargin < 5, 
    alpha = 1; 
end;

% simulating
[neto, OS, AH] = sim(net, IS, TS, dsteps, alpha);
MSE = sum(sum((TS(:,dsteps:end)-OS(:,dsteps:end)).^2)) ./ prod(size(OS(:,dsteps:end)));


% create output ESN object
neto = net;


