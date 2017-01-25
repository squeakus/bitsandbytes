NH  = 100;
NS  = 300;
NSR = 100;
PW  = 0.65;

DA = (0.5 * sin(pi / 10 * (1:2*NS)))';
DA = [DA, ones(2*NS, 1)];
D  = DA(1:NS,:);

WR = (2.0 * rand(NH, NH) - 1.0) .* (rand(NH, NH) < PW);
WR = WR ./ (max(abs(eig(WR))));
WR = WR .* 0.8;
WB = 2.0 * rand(NH, 2) - 1.0;
WB(:,2) = 0.0;

SI = 0.0;
ST = zeros(NS, NH);
ST(1,:) = SI;

for I = 2:NS,
    ST(I,:) = tanh(WR * ST(I-1,:)' + WB * D(I-1,:)' )';
end;

SO = [ST, ones(NS,1)];
%SO(NS,:) = [];
DO = D;
%DO(1,:) = [];
VO = SO(NSR:NS,:)\DO(NSR:NS,:);

OT = SO * VO;
O = zeros(NS,2);

STMSE = zeros(NS-NSR,1);
for I = 1:(NS-NSR),
    STMSE(I) = abs(OT(NSR+I,1) - D(NSR+I,1)) .^ 2 ;
end;

subplot(5,1,5), plot(STMSE);
sum(STMSE) / (NS-NSR+1),

STMSE = zeros(NS,1);
STT= zeros(NS, NH);
STT(NSR,:) = ST(NSR ,:);
O(NSR,:) = D(NSR, :);
O(NSR,:),
D(NSR,:),

%return;
for I = NSR+1:NS,
    STT(I,:) = tanh(WR * STT(I-1,:)' + WB * O(I-1,:)')';
    %STT(I,:) = ST(I,:);
    
    O(I,:) = [STT(I,:), 1] * VO;
    O(I,2) = 1;

    STMSE(I,1) = abs(O(I,1) - D(I,1)) .^ 2 ; %sum( (STT(I,:) - ST(I,:)) .^2) / NH;
    %O(I, :) = D(I, :);
end;

SO = [ST, ones(NS,1)];
OT = SO * VO;

subplot(5,1,1), plot(D);
subplot(5,1,2), plot(OT);
subplot(5,1,3), plot(O);
subplot(5,1,4), plot(STMSE);
