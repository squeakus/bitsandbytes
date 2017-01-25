function [X Y Z]=polybol3dstr(x1,y1,z1,x2,y2,z2,a,b,c,d)
xorg=1000000;
yorg=1000000;
zorg=1000000;
% x1(isnan(x1))=[];
% y1(isnan(y1))=[];
% z1(isnan(z1))=[];
if isempty(x1{1})==1
        X{1}=[];
        Y{1}=[];
        Z{1}=[];
elseif isempty(x2{1})==1
        X{1}=[];
        Y{1}=[];
        Z{1}=[];
else
for i=1:length(x1) 
    xv=x1{i};
    yv=y1{i};zv=z1{i};
    if isempty(xv)==0
    for j=1:length(xv)
    [xp yp zp]=mproject(xorg,yorg,zorg,xv(j),yv(j),zv(j),0,0,1,0);
    x1gt(j)=xp;
    y1gt(j)=yp;
    z1gt(j)=0;
    end
    x1g{i}=x1gt;y1g{i}=y1gt;z1g{i}=z1gt;
   clear x1gt y1gt z1gt
    else
    x1g{i}=[];y1g{i}=[];z1g{i}=[];
    end
    
    clear x1gt y1gt z1gt
end
for i=1:length(x2) 
    xv=x2{i};
    yv=y2{i};zv=z2{i};
    if isempty(xv)==0
    for j=1:length(xv)
    [xp yp zp]=mproject(xorg,yorg,zorg,xv(j),yv(j),zv(j),0,0,1,0);
    x2gt(j)=xp;
    y2gt(j)=yp;
    z2gt(j)=0;
    end
    x2g{i}=x2gt;y2g{i}=y2gt;z2g{i}=z2gt;
   clear x2gt y2gt z2gt
    else
    x2g{i}=[];y2g{i}=[];z2g{i}=[];
    end
    
    clear x2gt y2gt z2gt
end
[x1gc y1gc]=poly2cw(x1g,y1g);
[x2gc y2gc]=poly2cw(x2g,y2g);
[xint yint]=polybool('and',x1gc,y1gc,x2gc,y2gc);
if isempty(xint)~=1
    for i=1:length(xint)
    xreut=xint{i};yreut=yint{i};
    for j=1:length(xreut)
   [xp yp zp]=mproject(xorg,yorg,zorg,xreut(j),yreut(j),0,a,b,c,d);
   xret(j)=floor(xp*1000)/1000;
   yret(j)=floor(yp*1000)/1000;
   zret(j)=floor(zp*1000)/1000;
    end
   xreu{i}=xret;
   yreu{i}=yret;zreu{i}=zret;
   clear xret yret zret
end
X=xreu;
Y=yreu;
Z=zreu;
else
X{1}=[];
Y{1}=[];
Z{1}=[];
end
end
end