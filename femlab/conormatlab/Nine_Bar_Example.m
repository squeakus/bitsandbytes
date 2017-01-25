% clear memory

clear all

% Generation of coordinates and connectivities
% elementNodes represents node connectivites for bars

elementNodes=[ 1 2;2 3;3 4;4 5;5 6;6 1;1 3;3 5;3 6];
nodeCoordinates=[ 0 0;0 1;1 1;2 1;2 0;1 0];
nodeCoordinates_reset =[ 0 0;0 1;1 1;2 1;2 0;1 0];

% Calculating number of elements and nodes by evaluating
% size of corresponding matrices

numberElements = size(elementNodes,1);
numberNodes    = size(nodeCoordinates,1);

% E: modulus of elasticity
% A: area of cross section
% L: length of bar

E = 210*10^6; % kN/m^2

xx = nodeCoordinates(:,1);
yy = nodeCoordinates(:,2);


% for structure:
% displacements  - displacement vector
% force          - force vector
% stiffness      - stiffness matrix

GDof = 2*numberNodes;
U    = zeros(GDof,1);
force= zeros(GDof,1);


% Applying 10kN load for DOF = 12

force(12)= -100;

% Defining Fixities/Degrees of Freedom
prescribedDof = [1 2 10]';

% Calculating stresses with established areas
% [min_areas] = stresses2Dtruss(numberElements,elementNodes,xx,yy,displacements,E,initial_areas)

hold off
%-------------------------------------------------------------------------- 
pop = 150;               % size of sample estimations for each member area
iterations = 75;         % Number of iterations per phase (number of loops run)
limit = 1e-6;            % Mean of standard deviations that breaks the loop (accuracy)- CHECK THIS??????????????
k_fac_def = 1e15;         % Penalty Factor for displacement in Fitness Function
k_fac_area = 1e3;        % k value chosen to penalty to meaningful magnitude

                                                
best_areas = ones(1,numberElements);     % Initialising array of best areas
areas_best_in = zeros(1,numberElements); % initalisisng area_best_in as zeros array

i_mean = 0.01*ones(1,numberElements);        % means of areas (take inital min_areas as mean????)
i_std  = 0.1*i_mean;                    % variance of areas (0.1 is correct????)
i_mean_reset = 0.01*ones(1,numberElements); 
i_std_reset = 0.1*i_mean_reset;
std_check_limit = 1e-6;

                 
limit_pass = 0;                         % limit_pass is for convergence std_dev reset
max_limit_pass = 20;                    % Setting max. no of convergence resets of std. deviation (Varies with no. elements)
def_limit = 0.001;                      % Limiting deflection (say L/360) - where L = 2m (span length)

% -------------------------------------------------------------------------
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
    
% initialising area_stiff_in for use in stiffness m file. will be updated with each iteration   
        
    for i = 1:pop               % analysing areas for each set of areas in population
   
    area_in_stiff = areas(i,:); % Calling ith row of population to input into stiffness calcs    
       
%--------------------------------------------------------------------------
% for structure:
% displacements  - displacement vector
% force          - force vector
% stiffness      - stiffness matrix

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
                 
      delta_disp(i) = 0 - displacements(12);     % displacements(x) - where x is DOF of interest(ie DOF for applied load)
      def_fitness = def_limit - delta_disp(i);  % defining deflection fitness as difference of limit and actual
      
      if def_fitness < 0
          k_def = (1/(def_fitness))^6;    % Imposing harsh fitness penalty for deflections beyond limit
      else
          k_def = 0;                  % No Penalty for deflections within limit
      end % if def_fitness < 0
      
      
      for e = 1:numberElements       % For Loop ensures min_area criteria is adhered to by setting it to min. possible
          delta_areas(i,e) = areas(i,e) - min_areas(e); % Creating delta_areas(i,e) which is difference of min_area and actual
          if delta_areas(i,e)<0      % if areas are less than min_area allowed -> if function is triggered
              areas(i,e) = areas(i,e) + abs(delta_areas(i,e)); % adding difference in areas to keep at minimum allowed
          else
          end %if delta_areas(i,e)<0
      end % for e = 1:numberElements
            
      sum_of_vols(i) = sum(vol(i,:));
      
      
      % Performance Function to be maximised
      fitness(i) = -(sum_of_vols(i) + k_fac_def*k_def); 
                
                
         end % i = 1:pop
         
            % Find top 10% of answers
            N_answers = 0.1*pop;
            % Maximum fitness values and their indices at the top
            [fit ind] = sort(fitness,'descend'); %array places fittest solutions in decending order for calling
            
            for n = 1:N_answers
                fit_areas(n,:) = areas(ind(n),:);  % Best 10% of the area estimates from the population in this iteration
                
                if n==1
                    fittest_areas(flag,:) = areas(ind(n),:);  % Very best performing area estimate from the population in this iteration
                    fittest_area_plot(i,:) = areas(ind(n),:);                
                    fittest_sol(flag) = fit(1);                % Very best Performance Function value in this iteration
                                                           
                    flag = flag + 1                            % Prepare performance assignment for next iteration j
                end % n==1
            end % n = 1:N_answers
% Identify mean and standard deviation of the top 10% best performing areas 
% from this iteration and generate new sample for next iteration

% Break out of loop if standard deviation exceeds limit
 
for ij = 1:numberElements
     
    
    i_mean(ij)  = mean(fit_areas(:,ij));   % Mean of top 10% performing area estimates
    i_std(ij)   = std(fit_areas(:,ij));    % Standard Deviation of top 10% performing area estimates
 
     % if i == 1
if abs(mean(i_std)) < std_check_limit      % creating reset of standard deviation form current mean if convergence limit reached
   
    limit_pass  = limit_pass + 1;          % Adding to limit_pass to limit number of convergence resets
   
    if limit_pass > max_limit_pass
       
        areas(:,ij) = i_mean(ij) + i_std(ij)*randn(pop,1); % New sample for the next j-th iteration
   else
        
     areas(:,ij) = i_mean(ij) + i_std_reset(ij)*randn(pop,1); % i_std_reset variable resets stand. dev from mean
   
    end % abs(mean(i_std)) < limit

else % abs(mean(i_std)) < limit                
     
     areas(:,ij) = i_mean(ij) + i_std(ij)*randn(pop,1); % New sample for the next j-th iteration
   
    
end % abs(mean(i_std)) < std_check_limit

end % ij = 1:numberElements


% -------------------------------------------------------------
% View progress of Performance function
% -------------------------------------------------------------
         
figure(1)
plot(flag,delta_disp,'b+')
hold on
plot(flag,def_limit,'rx')
axis([0,iterations,0,0.002])      % Limiting axis to relevent values
xlabel('Iterations')
ylabel('Displacements (m)')
hold on

figure(2)
plot(flag,min_areas,'rx')
hold on
plot(flag, fittest_area_plot(i,:),'k+')
xlabel('Iterations')
ylabel('Fittest Areas (m^2)')
hold on

figure(3)
plot(j,sum_of_vols,'r*')
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
SUCCESS = XLSWRITE('D:\Users\Conor\Matlab Prog. File\bin\FILES FOR WRITE UP\2x1 NL Run8.xls',reshaped_best_areas,'Best_Areas')
SUCCESS = XLSWRITE('D:\Users\Conor\Matlab Prog. File\bin\FILES FOR WRITE UP\2x1 NL Run8.xls',reshaped_min_areas,'Min_Areas')


% Drawing displacements
% L=xx(2)-xx(1);

% ## OBTAINING FINAL SOLUTION FOR BEST AREAS ##

if j == iterations  % if j = iterations solve for fittest areas
% Computation of the system stiffness matrix
[stiffness]= formStiffness2Dtruss(GDof,numberElements,...
elementNodes,numberNodes,nodeCoordinates,xx,yy,E,best_areas);


displacements_best_areas = solution(GDof,prescribedDof,stiffness,force);

[Bar_Forces] = stresses2Dtruss_FinalRun(numberElements,elementNodes,...
xx,yy,displacements,E,best_areas)

SUCCESS = XLSWRITE('D:\Users\Conor\Matlab Prog. File\bin\FILES FOR WRITE UP\2x1 NL Run8.xls',Bar_Forces,'Bar Forces')

% output displacements/reactions
outputDisplacementsReactions(displacements,stiffness,...
GDof,prescribedDof);
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