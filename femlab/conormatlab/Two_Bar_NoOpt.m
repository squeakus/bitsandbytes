% ----------------------------------------------------------------------
% Conor Lyons - Adapted from Paraic Quirke (based on code by N.Harris 2007)
% 19th February 2014
% ----------------------------------------------------------------------
%
% Cross Entropy algorithm for Optimising member areas of truss in compression
%
% rng('shuffle'); % resets the random number generator

clear all
tic         % initiate the timer

% Generation of coordinates and connectivities
% elementNodes represents node connectivites for bars

% #### WATCH UNITS!!!!!!!!! #####
elementNodes=[1 2;2 3];
nodeCoordinates=[ 0 0;10 0;0 10];

% Calculating number of elements and nodes by evaluating
% size of corresponding matrices

numberElements = size(elementNodes,1);
numberNodes    = size(nodeCoordinates,1);

% E: modulus of elasticity
E = 210*10^6;

% Removed mass.m from code following mass additions to stresses2Dtruss.m

xx = nodeCoordinates(:,1);
yy = nodeCoordinates(:,2);

GDof = 2*numberNodes;
U    = zeros(GDof,1);
force= zeros(GDof,1);

% Applying 100kN load for DOF = 4
force(4)= -500;

% Defining Fixities/Degrees of Freedom
prescribedDof = [1 2 5 6]';

area_in_stiff = 100*1e-6*ones(1,numberElements)


%--------------------------------------------------------------------------
% for structure:
% displacements  - displacement vector
% force          - force vector
% stiffness      - stiffness matrix


% Computation of the system stiffness matrix
[stiffness]= formStiffness2Dtruss(GDof,numberElements,...
elementNodes,numberNodes,nodeCoordinates,xx,yy,E,area_in_stiff);

% Solution
displacements = solution(GDof,prescribedDof,stiffness,force);
us = 1:2:2*numberNodes-1;
vs = 2:2:2*numberNodes;

% Stresses at elements
[min_areas] = stresses2Dtruss(numberElements,elementNodes,...
xx,yy,displacements,E,area_in_stiff);

% output displacements/reactions
outputDisplacementsReactions(displacements,stiffness,...
GDof,prescribedDof);
% Following for loop to calculate element masses

                       
figure(1)
XX=displacements(us);YY=displacements(vs);
dispNorm=max(sqrt(XX.^2+YY.^2));
scaleFact=2*dispNorm;
clf
hold on
drawingMesh(nodeCoordinates+scaleFact*[XX YY],...
elementNodes,'L2','k.-');
drawingMesh(nodeCoordinates,elementNodes,'L2','k.--');
hold on



% End of Script 

