%% Gridpar
% in the form : Ax+By+C=0
% grid indicates the number of lines, must be odd number
% space indicates the distance between the lines
%% Input
%
%% Returns
%
%%
function[ag bg cg]=grid_par(xo,yo,theta,d,grid,space)
theta_rd=theta*pi/180;
m=tan(theta_rd);
a=-m;
b=1;
c1=-a*xo-yo+(d*sqrt(1+m^2));
c2=-a*xo-yo-(d*sqrt(1+m^2));
c=max(c1,c2);
c_shift=abs(space/cos(theta_rd));
co=c-((grid-1)/2)*c_shift;
cg_temp=0;
for i=1:grid
    cg(i)=co+(i*c_shift);
end
ag=a*ones(1,grid);
bg=b*ones(1,grid);
end
