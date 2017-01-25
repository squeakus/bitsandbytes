function [stiffness]=...
formStiffness2Dtruss_FinalRun(GDof,numberElements,...
elementNodes,numberNodes,nodeCoordinates,xx,yy,E,best_areas);
stiffness=zeros(GDof);
% computation of the system stiffness matrix

for e=1:numberElements;
% elementDof: element degrees of freedom (Dof)

indice=elementNodes(e,:);

% Finding DOF for ends of each bar
elementDof=[ indice(1)*2-1 indice(1)*2 indice(2)*2-1 indice(2)*2];


xa = xx(indice(2))-xx(indice(1));
ya = yy(indice(2))-yy(indice(1));

length_element = sqrt(xa*xa+ya*ya);

C = xa/length_element;
S = ya/length_element;

k1 = (E*area_in_stiff(:,e))/length_element*... %calling eth column of corresponding population row
[C*C C*S -C*C -C*S; C*S S*S -C*S -S*S;
-C*C -C*S C*C C*S;-C*S -S*S C*S S*S];



stiffness(elementDof,elementDof)=...
stiffness(elementDof,elementDof)+k1;
end