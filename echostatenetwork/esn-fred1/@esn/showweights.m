function showweights(net)
% Display weight matrix for recurrent weights

% check object
if ~ isa(net, 'esn'),
    error('Not an esn object.');
end;

figure('name','Weights');
hintonw(net.weightsRecurrent);
