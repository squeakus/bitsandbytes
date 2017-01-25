%% Optimizer
% This uses the inbuilt GA class and a fitness function to optimize input
%% Input
% None
%% Returns
% An optimized result and fitness value.
%%

function [best] = optimizer()
tic
% Show the default values of the genetic algorithm
options = gaoptimset(@ga);
options = gaoptimset(options, 'Generations', 10);
options = gaoptimset(options, 'PopulationSize', 50);
options = gaoptimset(options, 'PlotFcns', @gaplotbestf);
options

disp(['popsize:',num2str(options.PopulationSize)])
disp(['Generations:',num2str(options.Generations)])

%test optimisation function
%[x fval] = ga(@test_fitness, 6);
%test with lower and upper bounds
%lb = [0,0,0,0,0,0]
%ub = [200,200,200,200,200,200]
%[x,fval] = ga(@test_fitness, 6,[],[],[],[],lb,ub)

lb = [200,100,15,15];
ub = [400,500,75,90];

%[x fval, exitFlag,Output] = ga(@line_fitness, 4, [], [], [], [], lb, ub);
[x fval, exitFlag,Output] = ga(@line_fitness, 4, [], [], [], [], lb, ub,[],options);

disp('The best solution is');
disp(x);
disp('with a  fitness of');
disp(fval);
fprintf('The number of generations was : %d\n', Output.generations);
toc
