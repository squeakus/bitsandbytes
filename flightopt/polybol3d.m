%% Polybol3D
% Finds the projection of many polygons onto the ground and calculates the
% intersection
%% Input
% Polygon array, face array
%% Returns
%  Area
%% Example
%



function [X Y Z]=polybol3d(x1,y1,z1,x2,y2,z2,a,b,c,d)
xorg=1000000;
yorg=1000000;
zorg=1000000;
for i=1:length(x1)
    xv=x1(i);yv=y1(i);zv=z1(i);
    [xp yp zp]=mproject(xorg,yorg,zorg,xv,yv,zv,0,0,1,0);
    x1g(i)=xp;
    y1g(i)=yp;
    z1g(i)=zp;
end
for i=1:length(x2)
    xv=x2(i);yv=y2(i);zv=z2(i);
    [xp yp zp]=mproject(xorg,yorg,zorg,xv,yv,zv,0,0,1,0);
    x2g(i)=xp;
    y2g(i)=yp;
    z2g(i)=zp;
end
[x1gc y1gc]=poly2cw(x1g,y1g);
[x2gc y2gc]=poly2cw(x2g,y2g);
[xint yint]=polybool('and',x1gc,y1gc,x2gc,y2gc);
if size(xint)==0
    X=NaN;
    Y=NaN;
    Z=NaN;
else
    xno=x1(1);yno=y1(1);zno=z1(1);
    xn1=x1(2);yn1=y1(2);zn1=z1(2);
    xn2=x1(3);yn2=y1(3);zn2=z1(3);
    % [a,b,c,d]=plan3p(xno,yno,zno,xn1,yn1,zn1,xn2,yn2,zn2);
    for i=1:length(xint)
        [xp yp zp]=mproject(xorg,yorg,zorg,xint(i),yint(i),0,a,b,c,d);
        xret(i)=xp;
        yret(i)=yp;
        zret(i)=zp;
    end
    X=floor(xret*1000)/1000;
    Y=floor(yret*1000)/1000;
    Z=floor(zret*1000)/1000;
    % X=xret;
    % Y=yret;
    % Z=zret;
end
end