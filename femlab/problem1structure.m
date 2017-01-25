%................................................................
% MATLAB codes for Finite Element Analysis
% problem1Structure.m
% antonio ferreira 2008
% clear memory
clear all
% p1 : structure
p1=struct();
% elementNodes: connections at elements
p1.elementNodes=[1 2;2 3;2 4];
% GDof: total degrees of freedom
p1.GDof=4;
% numberElements: number of Elements
p1.numberElements=size(p1.elementNodes,1);
% numberNodes: number of nodes
p1.numberNodes=4;
% for structure:
% displacements: displacement vector
% force : force vector
% stiffness: stiffness matrix
p1.displacements=zeros(p1.GDof,1);
p1.force=zeros(p1.GDof,1);
p1.stiffness=zeros(p1.GDof);
% applied load at node 2
p1.force(2)=10.0;
% computation of the system stiffness matrix
for e=1:p1.numberElements;
% elementDof: element degrees of freedom (Dof)
elementDof=p1.elementNodes(e,:) ;
p1.stiffness(elementDof,elementDof)=...
p1.stiffness(elementDof,elementDof)+[1 -1;-1 1];
end
% boundary conditions and solution
% prescribed dofs
p1.prescribedDof=[1;3;4];
% solution
p1.displacements=solutionStructure(p1)
% output displacements/reactions
outputDisplacementsReactionsStructure(p1)