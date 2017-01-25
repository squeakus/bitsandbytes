clear classes;

IS = zeros(20, 4050);
TS = repmat(-0.5, 20, 4200);
for I=(0:20),
    IS(mod(I,20)+1, 200.*I + 1) = 0.5;
    TS(mod(I,20)+1, 200.*I + 1 : 200.*(I+1)) = 0.5;
end;
TS = TS(:,1:4050);
%TS = TS + rand(size(TS)) .* 0.05;

net = esn(20,100,20);
[net, me] = init(net, 0.02, -0.4, 1, -5, 0.2, -0.1);
net = notreshold(net);
net = setnotreshold(net);

disp(sprintf('Maximal eigenvalue is %f', me));
net = scaleeig(net, 0.44);

[net, MSETRN] = train(net, IS , TS, 50, 1.0);
disp(sprintf('Training error is %f', MSETRN));


IS_TEST = zeros(20,5000);
TS_TEST = zeros(20,5000);

for I=(1:20), IS_TEST(floor(rand .* 20)+1,floor(rand .* 5000)+1) = 0.5; end;
IS = [IS, IS_TEST];
TS = [TS, TS_TEST];

[net, MSETST, OS] = test(net, IS , TS, 4050);

disp(sprintf('Testing error os %f', MSETST));

for I=(1:20),
    subplot(20,1,I); plot(OS(I,:),'r');
end;
