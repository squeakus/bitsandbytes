%% Testfitness 
% This is a simplified fitness function for test matlab's genetic
% algorithm. We manually fix values for the best height, dist etc
%% Input
% Flight height,distance, flightline?, angle?, 
% spacing between lines?, number of lines
%% Returns
% fitness value, the smaller the better (like the shadow).
%%

function[fitness]=test_fitness(inputvector)
inputcell = num2cell(inputvector);
[fh,d,th1,th2,sg,ng] = inputcell{:};
fitness = 0;
fitness = fitness +  abs(300 - fh);
fitness = fitness +  abs(180 - d);
fitness = fitness +  abs(30 - th1);
fitness = fitness +  abs(15 - th2);
fitness = fitness +  abs(5 - sg);
fitness = fitness +  abs(5 - ng);
end
