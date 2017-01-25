%% Footprint:
% gives the convex hull of a set of points
%% Input:
% Set of points
%% Returns:
% plane
%%
function [kf]=footprint(cord)
xs=cord(:,1);ys=cord(:,2);
xs(isnan(xs))=[];ys(isnan(ys))=[];
cord=[xs ys];
kf=convhull(cord);
end