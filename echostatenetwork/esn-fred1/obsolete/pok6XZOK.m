clear classes;

%IS = zeros(0,4000);
%TS = TS + (2*rand(size(TS))-1) .* 0.0001;

[X, Y, Z] = create_lorenz(25000, 100);
X = X ./ 100;
Y = Y ./ 100;
Z = Z ./ 100;
IS = [Y];
TS = [X; Z];

net = esn(1, 100, 2);
[net, me] = init(net, 0.2, -0.1, 1.0, -0.5, 1.0, -0.5);
%(net, 0.0125, -0.4, 1, 0.20, 1, 0.20);
fprintf('Maximal eigenvalue is %.10f\n', me);
%net = scaleeig(net, 0.95);
net = settreshold(net, 0.0);

[net, MSETRN, OSTRN, AHTRN] = train(net, IS(:,1:15000) , TS(:,1:15000), 10000, 0.01, 0.01);
fprintf('Training error is %.10e\n', MSETRN);

rndlines = floor(rand(1,15) .* 100 + 1);
figure, plot(AHTRN(rndlines,:)');

[net, MSETST, OSTST, AHTST] = test(net, IS , TS, 10000, 0.01);
fprintf('Testing error os %.10e\n', MSETST);
figure, plot(OSTST(1,:));
figure, plot(OSTST(2,:));

%return;

%[net, MSETST, OSTST2, AHTST2] = test(net, IS , TS, 4000, 1.0);

snet = struct(net);
%figure, plot(TS); 
%figure, plot(OSTST); 
%figure, plot(OSTST2); 
%figure, plot(OSTST - OSTST2); 

%subplot(4,1,1);   plot(OS(1,1:end));
%subplot(4,1,2);   plot(X(1:end));
%subplot(4,1,3);   plot(Y(1:end));
%subplot(4,1,4);   plot(Z(1:end));

MAX_VALUE = max(max(abs(snet.weightsForward)));
VAL_CNT = sum(sum(abs(snet.weightsForward) > 0.1 * MAX_VALUE));
TOT_CNT = prod(size(snet.weightsForward));
NZ_CNT = sum(sum(snet.weightsForward ~= 0.0));
fprintf('Maximal value is %.10e\n', MAX_VALUE);
fprintf('No of values > 0.1 * MAX_VALUE is %d out of %d\n', VAL_CNT, TOT_CNT);
fprintf('No of nonzero values is %d out of %d\n', NZ_CNT, TOT_CNT);

