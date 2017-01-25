%% OnePlaneOneLine
% generates the shadow of a line on a plane 
%% Input
% array of points defining plane, a line
%% Returns
% 
%%

function [X Y Z xg yg]=oneplnoneln(shdx,shdy,shdz,a,b,c,d)
sizetem=size(shdx);
for i=1:sizetem(1)
    for j=1:sizetem(2)
        [xtem ytem ztem]=mproject(1000000,1000000,1000000,shdx(i,j),shdy(i,j),shdz(i,j),0,0,1,0);
        shdxg(i,j)=xtem;shdyg(i,j)=ytem;
    end
end
shdxg;
shdyg;
xun=[];yun=[];
for i=1:sizetem(1)
    [xtem ytem]=polybool('plus',xun,yun,shdxg(i,:),shdyg(i,:));
    xun=xtem;
    yun=ytem;
end
% xun
% yun
if isempty(xun)==1
    X=NaN;
    Y=NaN;
    Z=NaN;
    xg=NaN;
    yg=NaN;
else
    xg=xun;
    yg=yun;
    for i=1:length(xun)
        [xtem ytem ztem]=mproject(1000000,1000000,1000000,xun(i),yun(i),0,a,b,c,d);
        Xt(i)=xtem;Yt(i)=ytem;Zt(i)=ztem;
    end
    X=Xt;Y=Yt;Z=Zt;
end
end