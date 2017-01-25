function neto = scaleeig(net, new_eig)
% ESN.SCALEEIG - scale ESN's recurrent weights to given eigenvalue

max_eig = max(abs(eig(net.weightsRecurrent)));
net.weightsRecurrent = net.weightsRecurrent ./ max_eig .* new_eig;
neto = net;
