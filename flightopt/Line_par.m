function[a b c]=line_par(xo,yo,theta,d)
% to find the line parameters of the line with distance d from the point
% (xo,yo) and makes angle theta with the positive x axix
% in the form : Ax+By+C=0
theta_rd=theta*pi/180;
m=tan(theta_rd);
a=-m;
b=1;
c1=-a*xo-yo+(d*sqrt(1+m^2));
c2=-a*xo-yo-(d*sqrt(1+m^2));
c=max(c1,c2);
end
