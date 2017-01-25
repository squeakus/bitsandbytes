function [X Y Z]=polybol3dNaN(x1,y1,z1,x2,y2,z2,a,b,c,d)
xorg=1000000;
yorg=1000000;
zorg=1000000;
x1(isnan(x1))=[];
y1(isnan(y1))=[];
z1(isnan(z1))=[];
if isempty(x1)==1 
    X=x2;
    Y=y2;
    Z=z2;
elseif isempty(x2{1})==1
        X{1}=x1;
        Y{1}=y1;
        Z{1}=z1;
else
 for i=1:length(x1)
    xv=x1(i);yv=y1(i);zv=z1(i);
    [xp yp zp]=mproject(xorg,yorg,zorg,xv,yv,zv,0,0,1,0);
    x1gt(i)=xp;
    y1gt(i)=yp;
    z1gt(i)=zp;
    clear xv yv zv
 end
x1g{1}=x1gt;
y1g{1}=y1gt;z1g{1}=z1gt;
length(x2);
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
[xun yun]=polybool('union',x1gc,y1gc,x2gc,y2gc);
if isempty(xun)~=1
    for i=1:length(xun)
    xreut=xun{i};yreut=yun{i};
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