function neto = notreshold(net)
% 
% 
net.weightsInput(:,end) = 0;
neto = net;
