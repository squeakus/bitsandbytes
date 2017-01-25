function MGS = createmg(LEN, SUBS, ALPHA, BETA, GAMA, TAU, INITDUMMY)
% create Mackey-GLass series
%

if nargin < 2,
    SUBS = 10;
end;

if nargin < 3,
    ALPHA = 0.2;
end;

if nargin < 4,
    BETA = 10;
end;

if nargin < 5,
    GAMA = 0.1;
end;

if nargin < 6,
    TAU = 17;
end;

if nargin < 7,
    INITDUMMY = 200;
end;


start = SUBS * TAU + 1;
stop  = SUBS * (INITDUMMY + TAU + LEN + 1);
delay = SUBS * TAU;

MGS = zeros(1,LEN);
S = zeros(1,  stop);
S(start) = 1;

for SI=(start:stop-1),
    NOM1 = ALPHA * S(SI-delay);
    DEN1 = 1 + S(SI-delay)^BETA;
    S(SI+1) = S(SI) + (NOM1 / DEN1 - GAMA * S(SI)) / SUBS;
end;

S(1:SUBS * (INITDUMMY + TAU)) = [];

MGS = S(1:SUBS:SUBS*LEN);