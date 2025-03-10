function displacements=solution(GDof,prescribedDof,stiffness,force)

% function to find solution in terms of global displacements

% activeDOF represents DOF not restrained (ie no restraint present @ DOF)
activeDof = setdiff([1:GDof]',[prescribedDof]);

U = stiffness(activeDof,activeDof)\force(activeDof);

displacements = zeros(GDof,1);
displacements(activeDof)=U;