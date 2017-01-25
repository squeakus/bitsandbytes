function [X, Y, Z] = create_lorenz(seq_len, dummy_steps)
% CREATE_LORENZ creates time series of Lorenz attractor
% based on source on page "The Lorenz Attractor in 3D"
% http://astronomy.swin.edu.au/~pbourke/fractals/lorenz/
%
% seq_len      - length of desired sequence
% dummy_steps  - initial transient steps


H = 0.01;
A = 10.0;
B = 28.0;
C = 8.0 / 3.0;

total_len = seq_len + dummy_steps+1;
X = zeros(1, total_len);
Y = zeros(1, total_len);
Z = zeros(1, total_len);

X(1) = 0.1;
Y(1) = 0;
Z(1) = 0;
      
for I=(1:total_len-1),
    X(I+1) = X(I) + H * A*(Y(I)-X(I));
    Y(I+1) = Y(I) + H * (X(I)*(B-Z(I))-Y(I));
    Z(I+1) = Z(I) + H * (X(I)*Y(I)-C*Z(I) );
end;

X(1:dummy_steps+1) = [];
Y(1:dummy_steps+1) = [];
Z(1:dummy_steps+1) = [];