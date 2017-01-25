--TODO add grips to the grammar
--TODO default range for codon 1000?
--handle null strings better

--game stuff
local disqualification = false --tori gets no points if disqualified
local moved = false --only allows 1 move per individual
--file stuff
local name = "run" -- prefix for all the stats files
local grammarFile = "fightSimple.bnf" --BNF file to use
local statsFile = name.."0.dat" --stores info for each generation
local bestFile = name.."best0.txt"--stores the best of the current run
local popFile = "population.txt" --stores the current population
local bestFileExists = false --don't want to append to non existent file
local statsFileExists = false
--GA configuration
local crossOp = "onepoint" --specifies which crossover operator to use
local crossoverProb = 0.7 --probability of crossover occuring
local mutationProb =0.1 --probability of mutation occuring on a codon
local truncSize = 0.3 --% of population that gets through evaluation
local eliteSize = 2 --no. of individuals that survive each evaluation intact
local maxRuns = 50  --How many runs do you want it to do?
local maxGens = 100 --max number of Gens before reset
local popSize = 50 --how many individuals in your population
local chromeLength =22 --20 values for the joints + 2 for the grip
local maxChromeSize = 200
local maxWraps = 2
--GA Stuff
local initialised = false --checks if population has already been initialised
local generation = 0 --tracks no. of generations
local currentRun = 0 --how many times to run the experiment
local currentIndiv = 1 --tracks current individual
local individual ={fitness = 0, genome = {}, jointVals = {}, usedCodons = 0}
local bestPop = {} --stores the best o the population
local population = {} --array storing individuals
local bestIndiv ={}
--grammar stuff
local non_terminals = {}
local terminals = {}
local rules = {}
local start_rule = ""
--math.randomseed(os.time())

--utility methods
local function open_file(filename)
        local oldinput = io.input()
	io.input(filename)
	local file = io.input()
	io.input(oldinput)
	return file
     end

--[[ the new boss of operations, this parses a grammar file 
to generate the rules array. The individuals chromosome then 
selects from those rules to generate moves
--]]
local function read_bnf_file()
   NT = "NT"
   T = "T"
   local non_term_pattern = "[<>]"
   local rule_separator = "::="
   local separator = "|"
   
   local tmpFile = open_file(grammarFile)
   print("reading file:",grammarFile)
   while true do
      local line = tmpFile:read('*l')
      --initialising the production list for each rule
      local production_table = {}
      if not line then break end
      
      -- check for comments and empty lines
      if not(string.sub(line,1,1) == '#' or trim(line) == "") then
	 -- if its a rule, break it int left and right parts
	 if not (string.find(line, rule_separator) == nil) then
	    local lhs_end, rhs_start = string.find(line, rule_separator)
	    local lhs = trim(string.sub(line,1,lhs_end-1))
	    local rhs = string.sub(line,rhs_start+1,-1)
	    
	    -- make sure its NT and add it to the table
	    if not string.match(lhs,non_term_pattern) then
	       error("the lhs of a rule is not a non-terminal")
	    else
	       table.insert(non_terminals,{value = lhs,type = NT})
	       --initialise start rule
	       if start_rule=="" then
		  start_rule = {value = lhs, type = NT}
	       end
	    end
	    --strip out productions from rule
	    for production in rhs:gmatch("[^|]*[^|,\n]") do 
	       --if multiple parts, append it to tmp_production
	       local tmp_production = {}
	       production = trim(production)
	       if not string.match(production,non_term_pattern) then
		  table.insert(terminals,{value = production, type = T})
		  table.insert(tmp_production,{value = production,type = T})
	       else
		  --matches mixed types in a production(eg; <func> x)
		  production = string.gsub(production,"<[%a_]*>"," %1 ")
		  for value in production:gmatch("[%S+]*[%S]") do
		     if value:match(non_term_pattern) then
			table.insert(tmp_production,{value = value,type = NT})
		     else
			table.insert(tmp_production,{value = value,type = T})
		     end
		  end
	       end
	       table.insert(production_table,tmp_production)
	    end
	    if not rules[lhs] then
	       rules[lhs] = production_table	       
	    else
	       error("the lhs of each rule MUST be unique")
	    end
	 else
	    error("Each rule must be on one line")
	 end
      end
   end
end

local function map_chromosome(input)
   local cur_input = 0
   local current_symbol = {}
   local incremented = false
   local wraps = 0
   local output = {}
   local loopCounter =0
   local unexpanded_symbols = {start_rule}
 
   while wraps < maxWraps and table.getn(unexpanded_symbols) >0 do
      --expand wraps
      if cur_input % table.getn(input) == 0 and incremented then
	 wraps = wraps +1
	 incremented =  false
      end

      current_symbol = table.remove(unexpanded_symbols)
      --add to output, if it's a terminal
      if current_symbol.type == T then
	 table.insert(output,current_symbol.value)
      else
	 local production_choices = rules[current_symbol.value]
	 local prod_idx = input[(cur_input % table.getn(input))+1]
	 prod_idx = (prod_idx%table.getn(production_choices))+1
	 
	 if table.getn(production_choices) > 1 then
	    incremented = true
	    cur_input = cur_input+1
	 end

	 --remember! left to right (depth first) derivation
	 local current_production = production_choices[prod_idx]
	 for i =table.getn(current_production),1,-1 do
	    table.insert(unexpanded_symbols,current_production[i])
	 end
      end
   end
   if table.getn(unexpanded_symbols) >0 then
      print("not enough codons, incomplete mapping")
      return nil, cur_input
   else
      output = table.concat(output," ")
   end   
   return output, cur_input
end

function create_individual(chromosome,fitVal)
   chromosome = chromosome or {}
   fitVal =  fitVal or 0
   if table.concat(chromosome) == "" then
      local chrome_size = math.random(10,maxChromeSize)
      for i = 1,chrome_size do
	 table.insert(chromosome, math.random(1,1000))
      end
   end
   local tmpVals,tmpCodons = map_chromosome(chromosome)
   local tmp_individual = {fitness = fitVal,genome = chromosome,jointVals = tmpVals, usedCodons = tmpCodons}
   return tmp_individual       
end

--tidies the strings(from PiL2 20.4, thanks PiL!)
function trim(s) 
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

--[[ The fitness function is what drives the algorithm. I kept it
simple to start with for debugging reasons but you can add in whatever
you want(distance moved, decapitation,dismemberment,etc). If you make
a good one then post it one the forum!
--]]
local function get_fitness(wt)
   local tori_score = math.floor(get_player_info(1).injury)
   local uke_score = math.floor(get_player_info(0).injury)
   local fitness = tori_score - uke_score
   --this awards a fitness of zero if tori is disqualified
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
local function print_individual(indiv) 
   print("fit: "..indiv.fitness.. " val: "..indiv.jointVals.." used:"..indiv.usedCodons) 
end

-- GA functions
local function print_pop(pop)
   for i=1,table.getn(pop) do 
      print_individual(pop[i]) 
   end
end

local function write_pop()
   currentPop = write_file(popFile)
   for i=1,table.getn(population) do
      local tmpJointVals =population[i].jointVals or "invalid"
      currentPop:write(population[i].fitness,":",tmpJointVals,"\n") 
   end
   currentPop:close()
end

local function sort_pop()
   table.sort(population,function(a,b) return a.fitness > b.fitness end)
end

local function check_best()  
   if population[1].fitness > bestIndiv.fitness then
      bestIndiv = create_individual(population[1].genome,population[1].fitness)
      print("New best: "..bestIndiv.fitness.." "..bestIndiv.jointVals)   
      indivString = bestIndiv.fitness..": "..bestIndiv.jointVals
      if bestFileExists == false then	 
	 tmpFile = write_file(bestFile)
	 tmpFile:write(indivString.."\n")
	 tmpFile:close()
	 bestFileExists = true
      else
	 append_to_file(bestFile,indivString)
      end
   end
end

--[[
Mutate the individual by randomly chosing a new 
int with probability p_mut. Works per-codon.
--]]
local function intflip_mutation(chromosome) 
     for i=1,table.getn(chromosome) do
        if math.random() < mutationProb then
            chromosome[i] = math.random(1,1000)
	end
     end
     newIndividual = create_individual(chromosome)
    return newIndividual
end

--method to help with crossover
local function tableMerge(target,source,a,b)
   for i = a,b do
      table.insert(target,source[i])
   end
end

--[[
Given two individuals, create two children 
using one-point crossover and return them.
--]]
function onepoint(p, q, within_used)
   within_used = within_used or true
   local maxp, maxq = 0,0
   local pc,qc = p.genome, q.genome
   local c,d ={},{}

   if within_used == true then
      maxp, maxq = p.usedCodons, q.usedCodons
   else
      maxp, maxq = table.getn(pc), table.getn(qc)
   end

   local pt_p, pt_q = math.random(1,maxp), math.random(1,maxq)

   --split and recombine the genomes
   if math.random() < crossoverProb then
      tableMerge(c,pc,1,pt_p)
      tableMerge(d,qc,1,pt_q)
      tableMerge(c,qc,(pt_q+1),table.getn(qc))
      tableMerge(d,pc,(pt_p+1),table.getn(pc))
   else
      c,d = pc,qc
   end
   return create_individual(c), create_individual(d)
end

local function trunc_selection(pop)
   cutoff = math.floor(table.getn(pop)*truncSize)
   newPop = {}
   for i=1,cutoff do
      tmp_individual = create_individual(pop[i].genome)
      tmp_individual.fitness = pop[i].fitness
      table.insert(newPop,tmp_individual)
   end
   return newPop
end

local function generational_replace(bestPop)
   newPop = {}
   for i=1,eliteSize do
      individual = create_individual(bestPop[i].genome,bestPop[i].fitness)
      table.insert(newPop,individual)
   end
   newPopsize = table.getn(newPop)
   bestPopsize = table.getn(bestPop)
   while newPopsize < popSize do
      ind1, ind2 = math.random(1,bestPopsize), math.random(1,bestPopsize)
      newInd1, newInd2 = _G[crossOp](bestPop[ind1],bestPop[ind2],true)
      if newPopsize < popSize-1 then
	 table.insert(newPop,newInd1)
	 table.insert(newPop,newInd2)
      else
	 table.insert(newPop,newInd1)
      end
      newPopsize =table.getn(newPop)
   end   
    for i = eliteSize+1,newPopsize do
       newPop[i] = intflip_mutation(newPop[i].genome)
    end
    return newPop
end

--[[
creates a population of popsize, initialising 
 to 0 fitness and random jointVals
--]]
local function init_pop()
   print("initialising population")
   if next(rules) == nil then
      print("empty rule table, reading rules from file")
      read_bnf_file()
   end

   bestIndiv = create_individual()
   population = {}
   for i=1,popSize do 
      individual = create_individual()
      table.insert(population,individual)
   end
   --for index,table in pairs(population) do print("idx: "..index.." fitness: "..table.fitness.." jointvals: "..table.jointVals.." used: "..table.usedCodons)end
   initialised = true
end

local function get_average_fitness(pop)
   totalFitness =0
   for i=1,table.getn(pop) do
      totalFitness = totalFitness + pop[i].fitness
   end
   averageFitness = totalFitness/popSize
   return averageFitness
end

local function write_stats(statsFile)
   genStats = generation.." "..bestIndiv.fitness.." "..get_average_fitness(population) 
   print("Stats: "..genStats)
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
local function make_move(moveString)
   print(moveString)
   MOVE = {}
   --initialise tori so that he stands up
   for k,v in pairs(JOINTS) do
      set_joint_state(0, v, 3)
   end

   if moveString ~= nil then
      local f = assert(loadstring(moveString))
      f()--assigns limb values to variable MOVE
   
      for joint,value in pairs(MOVE) do 
	 set_joint_state(0, JOINTS[joint], value)
      end
   else
      print("***********************mapping incomplete***********************")
   end

--   set_grip_info(0, BODYPARTS.L_HAND,valArray[jointIndex+1]%3)
--   set_grip_info(0, BODYPARTS.R_HAND,valArray[jointIndex+2]%3)
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
local function next_game(wt)
   fitness = get_fitness(wt)
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
	    init_pop()
	 end	 
	 write_stats(statsFile)
      else
	 print("completed set no. of runs")
	 run_cmd("quit")
      end
      bestPop = trunc_selection(population)
      population = generational_replace(bestPop)
   end
   start_new_game()
end

local function start()
   next_turn()
end

add_hook("enter_freeze","keep stepping",next_turn)
add_hook("end_game","start next game",next_game)
add_hook("new_game","start",start)

