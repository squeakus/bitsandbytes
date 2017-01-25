% training ESN on multiple attractor behaviour

% create training input and output time series
IS = zeros(20, 4000);

TS = repmat(-0.5, 20, 4000);
for i=(0:19),
    IS(i+1, 200*i + 1) = 0.5;                % one input pike
    TS(i+1, 200*i + 1 : 200*(i+1)) = 0.5;    % output keeps value
end;

% create ESN object, 20 input, 200 hidden and 20 output units
net = esn(20,100,20);

% initialize ESN and display maximal eigenvalue and then scale it to 0.44
% 2% of recurrent weights set to 0.4 or -0.4
% 100% of input weights set to -5 or 5
% 20% of backward weights set to 0.1 or -0.1
% no treshold 
[net, me] = init(net, 0.02, -0.4, 0.2, -5, 0.2, -0.1);
net = settreshold(net, 0);

% scale recurrent weights eigenvalue and print old one
fprintf('Maximal eigenvalue before scaling to 0.44 is %f.\n', me);
net = scaleeig(net, 0.44);

% train ESN on given time series, supress 50 initial steps from training,
% diplay MSE of training phase
[net, MSETRN] = train(net, IS , TS, 50, 0.001);
fprintf('Training error is %.20f.\n', MSETRN);

% create testing input and output time series
% create 20 randomly positioned peaks and append them to the training
% sequence
IS_TEST = zeros(20,4000);

for i=(1:20), IS_TEST(ceil(rand .* 20),ceil(rand .* 4000)) = 0.5; end;
IS_TEST(find(sum(IS_TEST) > 0.5), :) = 0;
IS = [IS, IS_TEST];

TS = IS;
for i=(2:size(TS, 2)),
    if sum(TS(:,i)) == 0, TS(:,i) = TS(:,i-1); end;
end;
TS(find(TS == 0)) = -0.5;

% test ESN on the same time series, supress 4000 initial (training) steps 
% from MSE calculation
[net, MSETST, OS] = test(net, IS , TS, 1000);

% print MSE and plot output signal
fprintf('Testing MSE is %.20f.\n', MSETST);
for i=(1:20),
    subplot(20,1,i); plot(OS(i,:),'r'); hold on; plot(IS(i,:),'b');
end;









