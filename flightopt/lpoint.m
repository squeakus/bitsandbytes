%% Line point: 
% Calculate the projection of any point on a perpendicular? line
%% Input: 
% point coordinates and two line coordinates
%% Returns: 
% coordinate
%%
function [x,y]=lpoint(a,b,c,xo,yo)
x=floor((((a*c)+(b*(-b*xo+a*yo)))/(-a^2-b^2))*1000)/1000;
y=floor(((b*c+a*b*xo-a*a*yo)/(-a^2-b^2))*1000)/1000;
end