function [a,b,c,d]=plan3p(xo,yo,zo,x1,y1,z1,x2,y2,z2)
p=[xo yo zo];
q=[x1 y1 z1];
r=[x2 y2 z2];
pq=q-p;
pr=r-p;
n=cross(pq,pr);
a=floor((n(1))*1000)/1000;
b=floor((n(2))*1000)/1000;
c=floor((n(3))*1000)/1000;
d=floor(((-a*xo)-(b*yo)-(c*zo))*1000)/1000;
end