function [Bar_Forces] = stresses2Dtruss_FinalRun(numberElements,elementNodes,...
xx,yy,displacements_best_areas,E,best_areas)

% stresses at elements

for e=1:numberElements
    
indice=elementNodes(e,:);

elementDof=[ indice(1)*2-1 indice(1)*2 indice(2)*2-1 indice(2)*2];
xa=xx(indice(2))-xx(indice(1));
ya=yy(indice(2))-yy(indice(1));

length_element = sqrt(xa*xa+ya*ya);

C=xa/length_element;
S=ya/length_element;

sigma(e)=E/length_element* ...
[-C -S C S]*displacements_best_areas(elementDof);


end


% adding same formating as found in displacements and mass m-files for ease
% of identifying members

%#######--HIDING FOR COMPUTATIONAL EFFICIENCY--############
disp('stresses')
tt=1:numberElements; format
[tt' sigma']

n = numberElements;
%reshaping length matrix into vector to multiply by [A]
reshaped_best_areas = reshape(best_areas,n,1);


Bar_Forces = reshaped_best_areas.*(sigma'); % multiplying e-th column (which represents e-th member area)
                                               % by corresponding stresses




