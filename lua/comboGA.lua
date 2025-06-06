--GA configuration
local moves = 3                   --number of moves you want him to make
local popSize = 10                --how many individuals in your population
local crossoverProb = 0.7         --probability of crossover occuring
local mutationProb =0.1           --probability of mutation occuring on a codon
local truncSize = 0.3             --% of population that gets through round
local eliteSize = 2               --no. of individuals that survive each round intact
--games stuff
local round = 0 
local lastFrame = 0
local currentFrame =0             --counter to track frames
local moveFrame = 1000            --make a move after a certain amount of turns
--file stuff
local loadPrev = false              --load population stored in file from previous run
local bestFile = "best.txt"       --stores the best of the current run
local popFile = "population.txt"  --stores the current population
local fileExists = false          --don't want to append to non existent file
--GA stuff
local initialised = false         --checks if population has already been initialised
local chromeLength =22 * moves    --20 values for the joints + 2 for the grip
local jointIndex =0
local jointVals = {}
local bestIndiv ={}
local individual ={}              --tmp individual storage
local bestPop = {}                --stores the best o the population
local current_indiv = 1           --tracks current individual
local population = {}             --array storing individuals

--math.randomseed(os.time())

-- initialising best individual
local tmpVals={}
for i = 1 , chromeLength do
   table.insert(tmpVals,1)
end
bestIndiv = {fitness=0,jointVals= tmpVals}

--[[ The fitness function is what drives the algorithm. I kept it
simple to start with for debugging reasons but you can add in whatever
you want(distance moved, decapitation,dismemberment,etc). If you make
a good one then post it one the forum!
--]]
local function get_fitness()
   tori_score = math.floor(get_player_info(1).injury)
   uke_score = math.floor(get_player_info(0).injury)
   fitness = tori_score - uke_score
   return fitness
end

--utility methods
local function open_file(filename)
        local oldinput = io.input()
	io.input(filename)
	local file = io.input()
	io.input(oldinput)
	return file
     end

local function write_file(filename)
	local oldoutput = io.output()
	io.output(filename)
	local file = io.output()
	io.output(oldoutput)
	return file
     end


local function append_to_file(filename,string)
   local tmpInput = open_file(filename)
   local lines = {}
   
   while true do
        line = tmpInput:read("*l")
        if not line then break end
	table.insert(lines,line)
   end
   table.insert(lines,string)
    
   local tmpOutput = write_file(filename)
   for i=1,table.getn(lines) do
      tmpOutput:write(lines[i],"\n")
   end
   tmpInput:close()
   tmpOutput:close()
end

-- GA functions
local function print_pop(pop)
   for i=1,table.getn(pop) do 
      print(i.." fitness "..pop[i].fitness,"value: "..table.concat(pop[i].jointVals,",")) 
   end
end

local function write_pop()
   currentPop = write_file(popFile)
   for i=1,table.getn(population) do
      currentPop:write(population[i].fitness,":",table.concat(population[i].jointVals,","),"\n") 
   end
   currentPop:close()
end

local function sort_pop()
   table.sort(population,function(a,b) return a.fitness > b.fitness end)
end

local function check_best()  
   if population[1].fitness > bestIndiv.fitness then
      bestIndiv.fitness,bestIndiv.jointVals = population[1].fitness,population[1].jointVals
      print("New best individual: "..bestIndiv.fitness.." move: "..table.concat(bestIndiv.jointVals,","))   
      indivString = bestIndiv.fitness..": "..table.concat(bestIndiv.jointVals,",")
      if loadPrev == false and fileExists == false then	 
	 tmpFile = write_file(bestFile)
	 tmpFile:write(indivString.."\n")
	 tmpFile:close()
	 fileExists = true
      else
	 append_to_file(bestFile,indivString)
      end
   else
      population[1].fitness,population[1].jointVals = bestIndiv.fitness,bestIndiv.jointVals
   end
end

--[[
Mutate the individual by randomly chosing a new 
int with probability p_mut. Works per-codon.
--]]
local function intflip_mutation(individual) 
    for i=1,chromeLength do
        if math.random() < mutationProb then
            individual[i] = math.random(1,4)
	end
     end
    return individual
end

--[[
Given two individuals, create two children 
using one-point crossover and return them.
--]]
local function onepoint_crossover(p, q)
    point = math.random(1,chromeLength)
    c,d = {},{}
    if math.random() < crossoverProb then
       for i=1,point do
	  table.insert(c,p[i])
	  table.insert(d,q[i])
       end
       for i=point,chromeLength do
	  table.insert(d,p[i+1])
	  table.insert(c,q[i+1])
       end
    else
       for i=1,chromeLength do
	  table.insert(c,p[i])
	  table.insert(d,q[i])
       end
    end
    return c, d
end

local function trunc_selection(pop)
   cutoff = math.floor(table.getn(pop)*truncSize)
   newPop = {}
   for i=1,cutoff do
      individual = {fitness = pop[i].fitness,jointVals =pop[i].jointVals}
      table.insert(newPop,individual)
   end
   return newPop
end

local function generational_replace(bestPop)
   newPop = {}
   for i=1,eliteSize do
      individual = {fitness = bestPop[i].fitness,jointVals =bestPop[i].jointVals}
      table.insert(newPop,individual)
   end
 
   newPopsize = table.getn(newPop)
   bestPopsize = table.getn(bestPop)
   while newPopsize < popSize do
       if newPopsize < popSize-1 then
	  ind1, ind2 = math.random(1,bestPopsize), math.random(1,bestPopsize)	  
	  jVals1, jVals2 = onepoint_crossover(bestPop[ind1].jointVals,bestPop[ind2].jointVals)
	  newInd1 = {fitness = -1, jointVals = jVals1}
	  table.insert(newPop,newInd1)
	  newInd2 = {fitness = -1, jointVals = jVals2}
	  table.insert(newPop,newInd2)
	  newPopsize =table.getn(newPop)
       end
    end   
    for i = eliteSize+1,newPopsize do
       newPop[i].jointVals = intflip_mutation(newPop[i].jointVals)
    end
    return newPop
end


--[[
creates a population of popsize, initialising 
them to 0 fitness and random jointVals
--]]
local function init_pop()
   print("initialising population")
   population = {}
   for i=1,popSize do 
      valArray ={}
      for i=1,chromeLength do	 
	 table.insert(valArray,math.random(1,4))
      end
      individual = {fitness = 0,jointVals = valArray}
      table.insert(population,individual)
   end
   initialised = true
end

local function load_pop(fileName) 
   print("loading population from file")
   population ={}
   print_pop(population)
   tmpFile = open_file(fileName)

   for i=1,popSize do
      jVals ={}
      indivString = tmpFile:read("*l")
      jointStart = string.find(indivString,":")+1
      jointString = string.sub(indivString,jointStart)

      for val in jointString:gmatch("%w+") do
	     table.insert(jVals,val)
      end
      indiv ={fitness= 0, jointVals = jVals}
      table.insert(population,indiv)
   end
   
   initialised = true
   print_pop(population)
   tmpFile:close()
end

-- GAME FUNCTIONS!
-- takes an array of joint values and configures tori
local function make_move(valArray)
   jointVals = valArray
   for k,v in pairs(JOINTS) do 
      jointIndex = jointIndex +1
      set_joint_state(0, v, jointVals[jointIndex])
   end
   set_grip_info(0, BODYPARTS.L_HAND,jointVals[jointIndex+1]%3)
   set_grip_info(0, BODYPARTS.R_HAND,jointVals[jointIndex+2]%3)
   jointIndex = jointIndex +2
end

--This makes the game continuous and initialises population
local function next_turn()
   if initialised == false then
      if loadPrev == true then
	 load_pop(popFile)
      else
	 init_pop()
    end         
 end
  if currentFrame == 0 then
      make_move(population[current_indiv].jointVals)
   end
   currentFrame = (currentFrame+1) % moveFrame
   step_game()
end

--[[
This method runs at the end of the round. It runs the
GA after all individuals have fought 
--]]
local function next_game()
   fitness = get_fitness()
   print("indiv:"..current_indiv.." fitness: "..fitness.." value: "..table.concat(population[current_indiv].jointVals,","))
   population[current_indiv].fitness = fitness
   moved = false
   round = round + 1   
   echo("Round "..round..", move: "..current_indiv..", fitness:  "..fitness)
   current_indiv = current_indiv + 1
   jointIndex =0
   currentFrame =0
   --if everyone has fought, create next population   
   if(current_indiv > popSize) then
      for i=1,eliteSize do
	 if bestPop[1] ~= nil then  
	    if bestPop[i].fitness ~= population[i].fitness then
	       print("FUCK FUCK FUCK SOMETHING GONE WRONG: "..i)
	       print("old: "..bestPop[i].fitness.." new: "..population[i].fitness)
	       print(table.concat(bestPop[i].jointVals,","))
	       print(table.concat(population[i].jointVals,","))
	    end
	 end
      end

      sort_pop()
      check_best()
      write_pop()   
      bestPop = trunc_selection(population)
      population = generational_replace(bestPop)
      current_indiv = 1
   end
   start_new_game()
end

local function start()
   rules =get_game_rules()
   lastFrame = rules.matchframes - 50
   moveFrame = math.floor((lastFrame / rules.turnframes) /(moves-1))
   next_turn()
end

add_hook("enter_freeze","keep stepping",next_turn)
add_hook("end_game","start next game",next_game)
add_hook("new_game","start",start)

