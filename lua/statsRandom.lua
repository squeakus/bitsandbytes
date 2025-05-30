local bestFileExists = false      --don't want to append to non existent file
local initBest = true
local statsFileExists = false
local generation = 0              --tracks no. of generations
local currentRun = 0              --how many times to run the experiment
local name = "random"
local statsFile = name.."0.dat"    --stores info for each generation
local bestFile = name.."best0.txt"    --stores the best of the current run
local popFile = "population.txt"  --stores the current population
local moved = false               --only allows 1 move per individual
local disqualification =true      --award no points if player is disqualified 
local initialised = false         --checks if population has already been initialised
local individual ={}              --tmp individual storage
local bestPop = {}                --stores the best o the population
local currentIndiv = 1            --tracks current individual
local population = {}             --array storing individuals
local maxRuns = 30
local maxGens = 50                -- max number of Gens before reset
local popSize = 50                --how many individuals in your population
local chromeLength =22            --20 values for the joints + 2 for the grip
local jointVals = {}
local bestIndiv ={}
--math.randomseed(os.time())

-- initialising best individual
local function initialise_best()
   print("reinitialising best individual")
   tmpVals={}
   for i = 1 , chromeLength do
      table.insert(tmpVals,1)
      bestIndiv = {fitness=0,jointVals= tmpVals}
   end
   initBest = false
end
--[[ The fitness function is what drives the algorithm. I kept it
simple to start with for debugging reasons but you can add in whatever
you want(distance moved, decapitation,dismemberment,etc). If you make
a good one then post it one the forum!
--]]
local function get_fitness()
   tori_score = math.floor(get_player_info(1).injury)
   uke_score = math.floor(get_player_info(0).injury)
   fitness = tori_score - uke_score
   if disqualification then
      if wt == 2 then
	 local win = get_world_state().winner
	 if win~=-1 then
	    local winner = get_player_info(win).name
	    if winner == 'uke' then
	       fitness = 0
	    end
         end
      end
   end
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
      if bestFileExists == false then	 
	 tmpFile = write_file(bestFile)
	 tmpFile:write(indivString.."\n")
	 tmpFile:close()
	 bestFileExists = true
      else
	 append_to_file(bestFile,indivString)
      end
   else
      population[1].fitness,population[1].jointVals = bestIndiv.fitness,bestIndiv.jointVals
   end
end

--[[
creates a population of popsize, initialising 
them to 0 fitness and random jointVals
--]]
local function init_pop()
   print("initialising population")
   if initBest == true then
      initialise_best()
   end

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

local function get_average_fitness(pop)
   totalFitness =0
   for i=1,table.getn(pop) do
      totalFitness = totalFitness + pop[i].fitness
   end
   print("total fitness: "..totalFitness)
   averageFitness = totalFitness/popSize
   print("average fitness: "..averageFitness)
   return averageFitness
end

local function write_stats(statsFile)
   genStats = generation.." "..bestIndiv.fitness.." "..get_average_fitness(population) 
   print("Stats:"..genStats)
   print("in file:"..statsFile)
   if statsFileExists == false then
       tmpFile = write_file(statsFile)
       tmpFile:write(genStats.."\n")
       tmpFile:close()
       statsFileExists = true
   else
      append_to_file(statsFile,genStats)
   end
end

-- GAME FUNCTIONS!
-- takes an array of joint values and configures tori
local function make_move(valArray)
   jointVals = valArray
   jointIndex =0
   for k,v in pairs(JOINTS) do
      jointIndex = jointIndex +1
      set_joint_state(0, v, jointVals[jointIndex])
   end
   set_grip_info(0, BODYPARTS.L_HAND,jointVals[jointIndex+1]%3)
   set_grip_info(0, BODYPARTS.R_HAND,jointVals[jointIndex+2]%3)
end

--This makes the game continuous and initialises population
local function next_turn()
   if initialised == false then
      init_pop()
   end   
   if moved == false then
      make_move(population[currentIndiv].jointVals)
      moved = true
   end
   step_game()
end

--[[
This method runs at the end of every round. It runs the
GA after all individuals have fought 
--]]
local function next_game()
   fitness = get_fitness()
   print("Exp: "..name.." Run: "..currentRun.." Gen: "..generation.." Indiv:"..currentIndiv.." Fitness: "..fitness)
   population[currentIndiv].fitness = fitness
   moved = false  
   echo("Exp: "..name.." Run: "..currentRun..",Gen: "..generation..", move: "..currentIndiv..", fit:  "..fitness)
   currentIndiv = currentIndiv + 1

   --if everyone has fought, create next population   
   if(currentIndiv > popSize) then
      currentIndiv = 1
      generation = generation + 1
      sort_pop()
      check_best()
      write_pop()

      if currentRun <= maxRuns then
	 if generation > maxGens then
	    currentRun = currentRun +1
	    print("moving on to next run! new file names:")
	    statsFile = name..currentRun..".dat"
	    bestFile = name.."best"..currentRun..".txt"
	    statsFileExists = false
	    bestFileExists = false
	    print(statsFile)
	    print(bestFile)
	    generation =0
	    initBest = true
	    init_pop()
	 end	 
	 write_stats(statsFile)
      else
	 print("completed set no. of run")
	 run_cmd("quit")
      end
      init_pop()
   end
   start_new_game()
end

local function start()
   next_turn()
end

add_hook("enter_freeze","keep stepping",next_turn)
add_hook("end_game","start next game",next_game)
add_hook("new_game","start",start)

