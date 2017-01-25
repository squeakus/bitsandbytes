% clear memory
clear all

tic % Initialising counter

% generation of coordinates and connectivities
% elementNodes represents node connectivites for bars
elementNodes =[ 1 2;1 3;3 4;2 4;3 2;1 4;3 5;5 6;6 4;4 5;3 6];
nodeCoordinates =[ 0 0;0.5 0;0 0.5;0.5 0.5;0 1;0.5 1];
nodeCoordinates_reset =[ 0 0;0.5 0;0 0.5;0.5 0.5;0 1;0.5 1]; % reset for NL iterations

numberElements=size(elementNodes,1)
numberNodes=size(nodeCoordinates,1);

xx=nodeCoordinates(:,1);
yy=nodeCoordinates(:,2);

% E: modulus of elasticity

E = 210*1e6;

% for structure:
% displacements: displacement vector
% force : force vector
% stiffness: stiffness matrix
GDof=2*numberNodes;
U=zeros(GDof,1);
force=zeros(GDof,1);

% applied load at node 1
% Applying 1kN load for DOF = 1

force(10)=-1;
force(5) = 0.05; % Nominal Horz. Load to encourage buckling (10% of Vertical)

% Defining Fixities/Degrees of Freedom
prescribedDof=[1 2 9]';


%-------------------------------------------------------------------------- 
pop = 150;               % size of sample estimations for each member area
iterations = 300;        % Number of iterations per phase (number of loops run)
limit = 1e-6;            % Mean of standard deviations that breaks the loop (accuracy)- CHECK THIS??????????????
k_fac_def = 1e8;         % Penalty Factor for displacement in Fitness Function
k_fac_area = 1;          % initialising k value for penalty function
k_def_NL_fac = 1e11;     % initialising penalty factor if NL displacements non-converging
NL_Limit_Fac = 0.9;      % Percentage of reduced displacement desired for 3rd NL it. vs 2nd NL it.
                         % eg if want 3rd NL it. displacement increase to be no more
                         % than 50% of 2nd displacement increase than
                         % NL_Limit_Fac = 0.5
best_areas = ones(1,numberElements);     % Initialising array of best areas
areas_best_in = zeros(1,numberElements); % initalisisng area_best_in as zeros array

% NB Areas in m^2 -> 1e-5m^2 = 10mm^2
i_mean = 1e-4*ones(1,numberElements);    % means of areas (take inital min_areas as mean????)
i_std  = 0.1*i_mean;                     % variance of areas (0.1 is correct????)
i_mean_reset = 1e-4*ones(1,numberElements); 
i_std_reset = 0.2*i_mean_reset;
std_check_limit = 1e-8;
nl_plot = 5;                            % Initalising fig. counter for NL Plotting (starts at 5 at increases by 1 per plot)
areas_limit = 3e-7;                     % setting min areas so if lower converted to near zero value
areas_reduction = 0.95                  % Percentage reduction of areas if below limit 

limit_pass = 1;                         % limit_pass is for convergence std_dev reset - initially set to 1.
max_limit_pass = 75;                   % Setting max. no of convergence resets of std. deviation
def_limit = 1/2000;                     % Limiting deflection (cantilever: say L/360)
k_def_NL = 0;                           % Initalising NL penalty as zero value for null effect

NL_disp = zeros(pop,3);
NL_delta_disp = zeros(pop,3);
disp_plot = zeros(1,pop);
vol = zeros(pop,numberElements);
sum_of_vols = zeros(1,pop);
%-------------------------------------------------------------------------
% Initialise algorithm

 for ij = 1:numberElements
    
          % Generate area estimates based on mean and variance
          areas(:,ij) = i_mean(ij) + i_std(ij)*randn(pop,1);
            
              
    end % ij=1:numberElements  
    
flag = 1; % Performance assesment - Allows identification of number of iterations if error occurs?

for j = 1:iterations
    
     if flag > 2    % Creating if condition that the top performing solution from the previous iteration is brought forward as 1st member of population
        if pop == 1
        i_mean(ij) = fittest_areas(flag-1,ij);
        i_std(ij)  = 0;
        else
        end
     end
    
 rng shuffle   % Resetting random number generator 
% initialising area_stiff_in for use in stiffness m file. will be updated with each iteration   
        
    for i = 1:pop               % analysing areas for each set of areas in population
   
    area_in_stiff = areas(i,:); % Calling ith row of population to input into stiffness calcs    
       
%--------------------------------------------------------------------------
% for structure:
% displacements  - displacement vector
% force          - force vector
% stiffness      - stiffness matrix

nl = 1; % Initalising Non-Linear Iteration ( for nl = 1:3)

nodeCoordinates = nodeCoordinates_reset; % Reseting to original coordinates

for nl = 1:3

xx = nodeCoordinates(:,1);
yy = nodeCoordinates(:,2);

% Computation of the system stiffness matrix
[stiffness]= formStiffness2Dtruss(GDof,numberElements,...
elementNodes,numberNodes,nodeCoordinates,xx,yy,E,area_in_stiff);

% Solution
displacements = solution(GDof,prescribedDof,stiffness,force);
us = 1:2:2*numberNodes-1;
vs = 2:2:2*numberNodes;

% Stresses at elements
[min_areas] = stresses2Dtruss(numberElements,elementNodes,...
xx,yy,displacements,E,area_in_stiff);

% output displacements/reactions
outputDisplacementsReactions(displacements,stiffness,...
GDof,prescribedDof);
          
% Using permute & reshape function to convert matrix to vector for addition
% Q represents a vector of each original nodal coordinate
Q = permute( nodeCoordinates, [ 2 1 ] );
Q = reshape( Q,GDof,1);

% Adding vector of original coordinates Q to displacements vector
UpdatedCoords = Q + displacements;

% Reshaping Updated Cordinates to 2xGDof/2 sized matrix
UpdatedCoords = reshape(UpdatedCoords,2,(GDof/2));
UpdatedCoords = permute(UpdatedCoords,[2 1]);

      
% Updating nodeCoordinates to new deflected values for next iteration
nodeCoordinates = UpdatedCoords;


if nl == 1; % if loop to determine changes in displacement with each NL iteration
NL_disp(i,nl) = displacements(10); % Displacements at DOF of interest
disp_plot(i) = -displacements(10);
NL_delta_disp(i,nl) = abs(displacements(10));


    else if nl == 2;
        NL_disp(i,nl) = abs(displacements(10));
        NL_delta_disp(i,nl) = abs(NL_disp(i,nl)) - abs(NL_delta_disp(i,1));
        
    else if nl == 3;
         NL_disp(i,nl) = abs(displacements(10));
         NL_delta_disp(i,nl) = abs(NL_disp(i,nl)) - abs(NL_disp(i,2));
        end
         
      
        end  % else if nl == 2;
end % if nl == 1
if nl == 3 % if loop for nl not equal to one to check convergence penalty
NL_check = abs(NL_delta_disp(i,3)) - abs(NL_delta_disp(i,2));
    if NL_check < 0
            k_def_NL = 0;   
    else
        k_def_NL = (NL_check*1e3)^2      % else  penalty
         
         end % if NL_delta_disp_3 > NL_delta_disp_2
end % for nl ~= 1
format longg, NL_disp; % Refromatting decimals to long format for higher precision



% Following for loop to calculate element masses

for e = 1:numberElements
    
     %Calculating Element Lengths
     length = elementNodes(e,:);
     elementDof =[length(1)*2-1 length(1)*2 length(2)*2-1 length(2)*2] ;
     xa=xx(length(2))-xx(length(1));
     ya=yy(length(2))-yy(length(1));
    
     length_element(e) = sqrt(xa*xa+ya*ya); 
                        
     % Calc volume of each row/population by finding volume for each member
     % for corresponding members and member length.     
                        
     vol(i,e) = areas(i,e)*length_element(e); % Calculating volumes for each element column (densities are irrelevant)
     
     
     
    end % for e = 1:numberElements
                 
      delta_disp(i,nl) = 0 - displacements(10);     % displacements(x) - where x is DOF of interest(ie DOF for applied load)
      format long, delta_disp;
      def_fitness = def_limit - delta_disp(i);  % defining deflection fitness as difference of limit and actual
     
      if def_fitness < 0
          k_def = (1/(def_fitness))^2;    % Imposing harsh fitness penalty for deflections beyond limit
      else
          k_def = 0;                  % No Penalty for deflections within limit
      end % if def_fitness < 0   
      
      for e = 1:numberElements                                 % For Loop ensures min_area criteria is adhered to by setting it to min. possible
          delta_areas(i,e) = areas(i,e) - min_areas(e);        % Creating delta_areas(i,e) which is difference of min_area and actual
          if delta_areas(i,e)<0                                % if areas are less than min_area allowed -> if function is triggered
              k_area = (1/delta_areas(i,e))^2;
              areas(i,e) = areas(i,e) + abs(delta_areas(i,e)); % adding difference in areas to keep at minimum allowed
          else
              k_area = 1;
          end %if delta_areas(i,e)<0
      end % for e = 1:numberElements
      
      sum_of_vols(i) = sum(vol(i,:));
     
      % Performance Function to be maximised
      fitness(i) = -(sum_of_vols(i) + k_def_NL*k_def_NL_fac + k_fac_def*k_def); 
         
      nl = nl +1;  % Adding to non-linear iteration
end % for nl = 1:3 % End of non-linear iteration - Fitness of solution now evaluated
 
         end % i = 1:pop
         
            % Find top 10% of answers
            N_answers = 0.1*pop;
            % Maximum fitness values and their indices at the top
            [fit ind] = sort(fitness,'descend'); %array places fittest solutions in decending order for calling
            
            for n = 1:N_answers
                fit_areas(n,:) = areas(ind(n),:);          % Best 10% of the area estimates from the population in this iteration
                
                
                if n==1
                    fittest_areas(flag,:) = areas(ind(n),:);      % Very best performing area estimate from the population in this iteration
                    fittest_sol(flag) = fit(1);                  % Very best Performance Function value in this iteration
                    fittest_area_plot(i,:) = areas(ind(n),:);
                    NL_disp_plot = NL_delta_disp(ind(n),:) % Storing NL_delta_disp of fittest areas
                    
                    flag = flag + 1                              % Prepare performance assignment for next iteration j
                end % n==1
            end % n = 1:N_answers
            
 
% Identify mean and standard deviation of the top 10% best performing areas 
% from this iteration and generate new sample for next iteration

% Break out of loop if standard deviation exceeds limit


for ij = 1:numberElements   
    
    
    
    i_mean(ij)  = mean(fit_areas(:,ij));   % Mean of top 10% performing area estimates
    i_std(ij)   = std(fit_areas(:,ij));    % Standard Deviation of top 10% performing area estimates
      

if abs(mean(i_std)) < std_check_limit      % creating reset of standard deviation form current mean if convergence limit reached
     
    limit_pass  = limit_pass + 1;         % Adding to limit_pass to limit number of convergence resets
    
    
    if limit_pass > max_limit_pass              % Disregarding reset if max limit pass reached
       
        areas(:,ij) = i_mean(ij) + i_std(ij)*randn(pop,1); % New sample for the next j-th iteration
    else
        
        areas(:,ij) = i_mean(ij) + i_std_reset(ij)*randn(pop,1); % i_std_reset variable resets stand. dev from mean
        
                  
    end % abs(mean(i_std)) < limit

else % abs(mean(i_std)) < limit                
     
     areas(:,ij) = i_mean(ij) + i_std(ij)*randn(pop,1); % New sample for the next j-th iteration
   
    
end % abs(mean(i_std)) < std_check_limit

if j > iterations*0.5 % Limiting forced area change until after iterations*X%
if min_areas(:,ij) < areas_limit % setting if loop for min areas
                                 % if min area below limit (1e-7m^2 =
                                 % 0.01mm^2) then set area to almost zero
                                 % value (ie 1e-12m^2 = 1e-6mm)
                       
    areas(:,ij)= areas(:,ij)*areas_reduction; % reducing areas if min area below limit
    
end % if min_areas(:,ij) < areas_limit
end% if j > iterations

end % ij = 1:numberElements
     


% -------------------------------------------------------------
% View progress of Performance function
% -------------------------------------------------------------

% Plotting NL displacements for fittest sol every 10 iterations

figure(1)
plot(j,fitness,'b+') % Plotting for 1st displaced NL iteration
hold on      % Limiting axis to relevent values
axis([0,j,-10*1e-4,0])  
xlabel('Iterations')
ylabel('Fitness')
hold on

figure(2)
plot(j,min_areas,'rx')
hold on
plot(j,fittest_area_plot(i,:),'k+')
xlabel('Iterations')
ylabel('Best Areas (m^2)')
hold on


figure(3)
plot(j,sum_of_vols,'r*')
axis([0,j,0,0.001])               % Limiting axis to relevent values
xlabel('Iterations')
ylabel('Total Volume (m^3)')
hold on

            % Set criteria for convergence.
                       
        end  % j = 1:iterations
        
% Now identify very best performance value and very best performing
% area estimates
    

[fit ind] = sort(fittest_sol,'descend');
    
% Record best area values
    
best_areas = fittest_areas(ind(1),:);
min_areas;


n = numberElements;
reshaped_best_areas = reshape(best_areas,n,1) %reshaping for tabular output
reshaped_min_areas = reshape(min_areas,n,1)   %reshaping for tabular output

% Writing Results to Excel File
SUCCESS = XLSWRITE('D:\Users\Conor\Matlab Prog. File\bin\FILES FOR WRITE UP\2x1 NL RunFINAL.xls',reshaped_best_areas,'Best_Areas')
SUCCESS = XLSWRITE('D:\Users\Conor\Matlab Prog. File\bin\FILES FOR WRITE UP\2x1 NL RunFINAL.xls',reshaped_min_areas,'Min_Areas')


% Drawing displacements
% L=xx(2)-xx(1);

% ## OBTAINING FINAL SOLUTION FOR BEST AREAS ##

if j == iterations  % if j = iterations solve for fittest areas
    
nl = 1
for nl = 1:3

xx = nodeCoordinates(:,1);
yy = nodeCoordinates(:,2);
[stiffness]=...
formStiffness2Dtruss_FinalRun(GDof,numberElements,...
elementNodes,numberNodes,nodeCoordinates,xx,yy,E,best_areas);

% Solution
displacements = solution(GDof,prescribedDof,stiffness,force);
us = 1:2:2*numberNodes-1;
vs = 2:2:2*numberNodes;

[Bar_Forces] = stresses2Dtruss_FinalRun(numberElements,elementNodes,...
xx,yy,displacements,E,best_areas)



SUCCESS = XLSWRITE('D:\Users\Conor\Matlab Prog. File\bin\FILES FOR WRITE UP\2x1 NL RunFINAL.xls',Bar_Forces,'Bar Forces')

% output displacements/reactions
outputDisplacementsReactions(displacements,stiffness,...
GDof,prescribedDof);
          
% Using permute & reshape function to convert matrix to vector for addition
% Q represents a vector of each original nodal coordinate
Q = permute( nodeCoordinates, [ 2 1 ] );
Q = reshape( Q,GDof,1);

% Adding vector of original coordinates Q to displacements vector
UpdatedCoords = Q + displacements;

% Reshaping Updated Cordinates to 2xGDof/2 sized matrix
UpdatedCoords = reshape(UpdatedCoords,2,(GDof/2));
UpdatedCoords = permute(UpdatedCoords,[2 1]);

      
% Updating nodeCoordinates to new deflected values for next iteration
nodeCoordinates = UpdatedCoords;

FinalNLPlot(nl) = displacements(10);

format longg, FinalNLPlot % Formatting NL Plot of displacements to be long decimal value

nl = nl + 1
end 

end % if j == iterations

figure(4)
XX=displacements(us);YY=displacements(vs);
dispNorm=max(sqrt(XX.^2+YY.^2));
scaleFact=2*dispNorm;
clf
hold on
drawingMesh(nodeCoordinates+scaleFact*[XX YY],...
elementNodes,'L2','k.-');
drawingMesh(nodeCoordinates_reset,elementNodes,'L2','k.--'); % using reset coords to show original shape
hold on

toc % output time taken

% End of Script 