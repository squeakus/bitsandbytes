function [neto] = settreshold(net, treshold);
% ESN.SETTRESHOLD - set network treshold

% check object
if ~ isa(net, 'esn'),
    error('Not an esn object.');
end;

% set treshold value, if not given
if nargin < 2, 
    treshold = 0;
end;

% set treshold weights
net.weightsInput(:,end) = treshold;
neto = net;
