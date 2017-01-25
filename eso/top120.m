%%% A 120 LINES TOPOLOGY OPTIMIZATION CODE BY Guilian Yi, February 2012%%%
function top120(nelx,nely,udis,penal,rmin)
% INITIALIZE
x(1:nely,1:nelx) = 1.0;
w0(1:nely,1:nelx) = 1.0;
upperdis = udis;
loop = 0; 
change = 1.;
[D,De]=FE(nelx,nely,x,penal);
% START ITERATION
while change > 0.01  
   loop = loop + 1;
   xold = x;
% FILTERING DISPLACEMENT CONTRIBUTION 
   [De]=check(nelx,nely,rmin,x,De);
   D0=De.*(x.^penal);
% DETERMIN ACTIVE ARRAY AND PASSIVE ARRAY
   IACTN = D0(:,:)>=0;
% CALCULATE THE DESIGN VARIABLES
   IACVAR = 1; % MARK OF CHANGE IN IACTN
   while IACVAR ~= 0
       one=ones(nely,nelx);
       sumDW=sum(sum(IACTN.*(D0.^(1/(penal+1))).*(w0.^(penal/(penal+1)))));
       u0=sum(sum((one-IACTN).*D0./x.^penal));
       IACVAR = 0;
       nx=zeros(nely,nelx);
       for ely = 1:nely
           for elx = 1:nelx
               if IACTN(ely,elx)== 1
                   newx=(sumDW/(udis-u0))^(1/penal)*(D0(ely,elx)/w0(ely,elx))^(1/(penal+1));
                   if newx>=1.0
                       nx(ely,elx)=1.0;
                       IACTN(ely,elx) = 0;
                       IACVAR = 1;
                   elseif newx<=0.001
                       nx(ely,elx)=0.001;
                       IACTN(ely,elx) = 0;
                       IACVAR = 1;
                   else
                       nx(ely,elx)=newx;
                   end
               else
                   nx(ely,elx)=x(ely,elx);
               end
           end
       end
       x=nx;
   end
% FEM ANALYSIS FOR REAL DISPLACEMENT FOR FINAL SOLUTIONS
   [D,De]=FE(nelx,nely,x,penal);
   change = max(max(abs(x-xold))); 
   disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%9.4f',sum(sum(x))) ' D.: ' sprintf('%8.4f',D) ' UD.: ' sprintf('%8.4f',udis) ' ch.: ' sprintf('%6.3f',change)]);
   colormap(gray); imagesc(-x); axis equal; axis tight; axis off;pause(1e-6);
% UPDATING DISPLACEMENT UPPER BOUNDARY
   alpha = D/upperdis;
   udis = udis/alpha;
end
%%%%%%%%%%%%FILTER FUNCTION%%%%%%%%%%%%%%%%%%%%%%
function [D0n]=check(nelx,nely,rmin,x,D0)
D0n=zeros(nely,nelx);
for i = 1:nelx
  for j = 1:nely
    sum=0.0; 
    for k = max(i-round(rmin),1):min(i+round(rmin),nelx)
      for l = max(j-round(rmin),1):min(j+round(rmin),nely)
        fac = rmin-sqrt((i-k)^2+(j-l)^2);
        sum = sum+max(0,fac);
        D0n(j,i) = D0n(j,i) + max(0,fac)*x(l,k)*D0(l,k);
      end
    end 
    D0n(j,i) = D0n(j,i)/sum;
  end
end
%%%%%%%%%% FE-ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D,De]=FE(nelx,nely,x,penal)
   F = -1.0;
   [KE] = lk; %ELEMENT STIFFNESS
   K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
   RF = sparse(2*(nely+1)*(nelx+1),1); RU = sparse(2*(nely+1)*(nelx+1),1); 
   for ely = 1:nely
     for elx = 1:nelx
       n1 = (nely+1)*(elx-1)+ely; 
       n2 = (nely+1)* elx   +ely;
       edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
       K(edof,edof) = K(edof,edof) + x(ely,elx)^penal*KE;    %TOTAL STIFFNESS
     end
   end
   fixeddofs = union([1:2:2*(nely+1)],[2*(nelx+1)*(nely+1)]); % MBB BEAM
   RF(2,1) = F;
   alldofs = [1:2*(nely+1)*(nelx+1)];
   freedofs = setdiff(alldofs,fixeddofs);
   RU(freedofs,:) = K(freedofs,freedofs) \ RF(freedofs,:);      
   RU(fixeddofs,:) = 0.0;
   VU = RU./abs(F);
   D = 0.0;
   De(1:nely, 1:nelx) = 0.0;
   for ely = 1:nely
     for elx = 1:nelx
        n1 = (nely+1)*(elx-1)+ely; 
        n2 = (nely+1)* elx   +ely;
        Ke = x(ely,elx)^penal*KE; 
        RFe = Ke*RU([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1); 
        VUe = VU([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1);
        De(ely,elx) = RFe'*VUe; 
        D = D+De(ely,elx);
     end
   end
%%%%%%%%%% ELEMENT STIFFNESS MATRIX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [KE]=lk
E = 1.0;
nu = 0.3;
k = [ 1/2-nu/6   1/8+nu/8 -1/4-nu/12 -1/8+3*nu/8 ... 
     -1/4+nu/12 -1/8-nu/8  nu/6       1/8-3*nu/8];
KE = E/(1-nu^2)*[ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
                    k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
                    k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
                    k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
                    k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
                    k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
                    k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
                    k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)];

