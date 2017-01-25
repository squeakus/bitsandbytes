function [k]=outerbound(cord,fh)
% to find the index of the points representing the visible boundary of the
% building
xs=cord(:,1);ys=cord(:,2);zs=cord(:,3);xls=cord(:,4);yls=cord(:,5);
xs(isnan(xs))=[];ys(isnan(ys))=[];zs(isnan(zs))=[];xls(isnan(xls))=[];yls(isnan(yls))=[];
zls=fh*ones(length(xls),1);
cord=[xs ys zs xls yls zls];
f=convhull(xs,ys);
chkint=inpolygon(cord(:,4),cord(:,5),cord(f,1),cord(f,2));
chkinn=ismember(1,chkint);
if chkinn==1
for i=1:length(cord)
xorg=cord(i,4);
yorg=cord(i,5);
zorg=cord(i,6);    
[x y z]=mproject(xorg,yorg,zorg,cord(i,1),cord(i,2),cord(i,3),0,0,1,0);
cordo(i,:)=[x y z];
end
temm=[cordo(:,1) cordo(:,2)];
k=convhull(temm);
for j=1:length(xs)
    for jj=1:length(k)
    if xs(j)==xs(k(jj)) && ys(j)==ys(k(jj)) && zs(j)>zs(k(jj))
        k(jj)=j;
    end
    end
end


else
for i=1:length(cord)
xorg=cord(i,4);
yorg=cord(i,5);
zorg=cord(i,6);    
[x y z]=mproject(xorg,yorg,zorg,cord(i,1),cord(i,2),cord(i,3),0,0,1,0);
cordo(i,:)=[x y z];
end
temm=[cordo(:,1) cordo(:,2)];
k=convhull(temm);

end
end