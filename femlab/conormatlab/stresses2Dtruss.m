function [min_areas] = stresses2Dtruss(numberElements,elementNodes,...
xx,yy,displacements,E,area_in_stiff)

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
[-C -S C S]*displacements(elementDof);


end


% adding same formating as found in displacements and mass m-files for ease
% of identifying members

%#######--HIDING FOR COMPUTATIONAL EFFICIENCY--############
disp('stresses')
tt=1:numberElements; format
[tt' sigma']

n = numberElements;
%reshaping length matrix into vector to multiply by [A]
reshaped_area_in_stiff = reshape(area_in_stiff,n,1);


Bar_Forces = reshaped_area_in_stiff.*(sigma'); % multiplying e-th column (which represents e-th member area)
                                               % by corresponding stresses
disp('Bar Forces (kN)')                       % Hiding for computational efficiency
kk = 1:numberElements; format
[kk' Bar_Forces]


% ##################################################
% Minimum areas calculation due to stress & buckling
% ##################################################

max_stress = 150*10^3; % 150N/mm^2 -> 150000kN/m^2

for e = 1:numberElements
    
    %Calculating Element Lengths
    length = elementNodes(e,:);
    elementDof =[ length(1)*2-1 length(1)*2 length(2)*2-1 length(2)*2] ;
    xa=xx(length(2))-xx(length(1));
    ya=yy(length(2))-yy(length(1));
    
    length_element(e) = sqrt(xa*xa+ya*ya);
    
    % Calculating minimum radii for buckling for each element
    % r_min = ((4FL^2)/(pi^3*E))^(1/4)
    
    % finding positive force value for each member for calcs (sqrt(x^2))
    % ##### IS THIS STRESS ANALYSIS SUITED FOR TENSION? #####
    % ##### ANY MERIT TO CONVERTING TENSILE TO COMP FOR BUCKLING - LOAD
    % CASES ETC????? ############
    
    % -- could add: if force = negative -> skip buckling - etc...
    
    ftemp = abs(Bar_Forces(e));
    
    
    
    min_r_buck(e) = ((4*ftemp*length_element(e)^2)*(1/((pi()^3)*E)))^(1/4);
    min_area_buck(e) = pi()*(min_r_buck(e)^2); %min area for corresponding radius for buckling
    min_area_stress(e) = ftemp/max_stress;
    
    if Bar_Forces(e)<0 % if negative (compression) - then consider buckling
    min_areas(e) = max(min_area_buck(e),min_area_stress(e));
    
    else if Bar_Forces(e)>0  % if pos (tension) - then consider stress only
    min_areas(e) = min_area_stress(e);
    
        else if Bar_Forces == 0 % Setting min area to zero if zero force
                min_areas(e) = 0
            end % else if Bar_Forces == 0 
        end % else if Bar_Forces(e)>0
    end % if Bar_Forces(e)<0 
    end




