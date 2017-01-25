%................................................................
function outputDisplacementsReactionsStructure(p)
% output of displacements and reactions in
% tabular form
% GDof: total number of degrees of freedom of
% the problem
% displacements
disp('Displacements')
jj=1:p.GDof; format
[jj' p.displacements]
% reactions
F=p.stiffness*p.displacements;
reactions=F(p.prescribedDof);
disp('reactions')
[p.prescribedDof reactions]