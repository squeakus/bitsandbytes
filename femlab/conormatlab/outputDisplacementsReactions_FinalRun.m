%................................................................

function outputDisplacementsReactions...
    (displacements_best_areas,stiffness,GDof,prescribedDof)

% output of displacements and reactions in
% tabular form

% GDof: total number of degrees of freedom of 
% the problem

% displacements
disp('Final Displacements')
%displacements=displacements1; 
jj=1:GDof; format
[jj' displacements_best_areas]

% reactions
F=stiffness*displacements_best_areas;
reactions=F(prescribedDof);
disp('reactions')
[prescribedDof reactions]