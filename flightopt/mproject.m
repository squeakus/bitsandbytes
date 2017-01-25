function[xp yp zp]=mproject(xl,yl,zl,xv,yv,zv,nx,ny,nz,d)
n=[nx ny nz];
L=[xl;yl;zl];
uu=n*L;
MM=[(n*L)+d-(nx*xl) -ny*xl -nz*xl -d*xl;
    -nx*yl (n*L)+d-(ny*yl) -nz*yl -d*yl;
    -nx*zl -ny*zl (n*L)+d-(nz*zl) -d*zl;
    -nx -ny -nz n*L];
M=MM';
prv=[xv yv zv 1]*M;
xxp=prv(1);
yyp=prv(2);
zzp=prv(3);
wp=prv(4);
xp=floor((xxp/wp)*10000)/10000;
yp=floor((yyp/wp)*10000)/10000;
zp=floor((zzp/wp)*10000)/10000;
% xp=xxp/wp;
% yp=yyp/wp;
% zp=zzp/wp;
end