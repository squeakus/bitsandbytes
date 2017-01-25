%% ZVal
% Finds the height directly above a point
%% Input
% a point and a plane
%% Returns
% Coordinate
%%

function [z]=zval(x,y,xv,yv,zv)
for i=1:length(xv)-1
    x1=xv(i);
    y1=yv(i);
    z1=zv(i);
    x2=xv(i+1);
    y2=yv(i+1);
    z2=zv(i+1);
    %     ww=[x1 y1 z1];
    %     tt=[x2 y2 z2];
    
    if x==x1 && y==y1
        z=z1;
    elseif x==x2 && y==y2
        z=z2;
    elseif x1==x2
        if x==x1
            if min(y1,y2)<y && max(y1,y2)>y
                y1-y2;
                z=z1+(((z2-z1)*(y1-y))/(y1-y2));
            end
        end
    elseif y1==y2
        if y==y1
            if min(x1,x2)<x && max(x1,x2)>x
                z=z1+((z2-z1)*(x1-x))/(x1-x2);
                
            end
        end
        
    elseif ((x-x1)/(y-y1))==((x-x2)/(y-y2))
        if min(x1,x2)<x && max(x1,x2)>x
            z=z1+(z2-z1)*((sqrt((x-x1)^2+(y-y1)^2))/(sqrt((x2-x1)^2+(y2-y1)^2)));
        end
    end
end
end



%
% for j=(i+1):length(xv)
%     x1=xv(i);y1=yv(i);x2=xv(j);y2=yv(j);
%     if x==x1 && y==y1
%         z=zv(i)
%     elseif x==x2 && y==y2
%         z=zv(j)
%     elseif (x-x1/x-x2)==(y-y1/y-y2)
%          if min(x1,x2)<x &&max(x1,x2)>x
%              z=zv(i)+(((zv(j)-zv(i))*(xv(i)-x))/(xv(i)-xv(2)))
%          elseif min(y1,y2)<y &&max(y1,y2)>y
%              z=zv(i)+(((zv(j)-zv(i))*(yv(i)-y))/(yv(i)-yv(2)))
%          end
%     end
%    end