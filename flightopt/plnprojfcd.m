function[Xfp Yfp Zfp]=plnprojfcd(Xshd,Yshd,Zshd,Xlt,Ylt,Zlt,nx,ny,nz,d)
% to find the projection of a polygon bounded by Xshd,Yshd,Zshd
for i=1:length(Xshd)
[xp yp zp]=mproject(Xlt(i),Ylt(i),Zlt(i),Xshd(i),Yshd(i),Zshd(i),nx,ny,nz,d);
Xfp(i)=floor(xp*1000)/1000;
Yfp(i)=floor(yp*1000)/1000;
if zp<Zlt(i)
Zfp(i)=floor(zp*1000)/1000;
else
Zfp(i)=-zp;
end
end
end