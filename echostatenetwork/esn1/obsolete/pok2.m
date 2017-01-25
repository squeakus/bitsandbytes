clear classes;

NUMUNITSHIDDEN  = 400;
DUMMYSTEPS = 1000;

LAS = loadseq('laser.trs','BIN');
LAS = tanh(LAS);

IS = LAS(:,1:8000);
TS = LAS(:,2:8001);


net = esn(1,NUMUNITSHIDDEN,1);
net = init(net, 0.6, 0.02, 1.0, 1.0);

[net, MSE] = train(net, IS , TS, DUMMYSTEPS);
[net, MSE, OS] = test (net, IS , TS, DUMMYSTEPS);

MSE

%plot(OS(1,:),'r'); hold on;
%plot(OS(2,:),'g');