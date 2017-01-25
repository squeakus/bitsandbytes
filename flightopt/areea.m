%% Areea2: 
% Computes the area of a polygon in 3 dimensions
% Source Ronald Goldman, "Area of Planar Polygons and Volume of Polyhedra" in Graphics Gems II (1994)
%% Input: 
%point array, faces
%% Returns:
% ?
%%

function [A]=areea(X,Y,Z)
temp1=isnan(X);
if temp1==1
    A=0;
else
    Arae=[0;0;0];
    for i=1:(length(X)-1)
        Vi=[X(i);Y(i);Z(i)];
        Vi1=[X(i+1);Y(i+1);Z(i+1)];
        arr(:,i)=cross(Vi,Vi1);
        Arae=Arae+arr(:,i);
    end
    d1=[X(1);Y(1);Z(1)];
    d2=[X(2);Y(2);Z(2)];
    d3=[X(3);Y(3);Z(3)];
    n1=d2-d1;
    n2=d3-d1;
    nn=cross(n1,n2);
    nr=norm(nn);
    n=nn./nr;
    areea=dot(n,Arae)/2;
    A=abs(areea);
end
end