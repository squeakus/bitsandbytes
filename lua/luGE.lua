#!/usr/bin/lua
--[[
TODO:
find the fucking bug thats causing it to screw up
check tourney is working
check gen replace works with tourney
]]--

--GA configuration
local target = "abababababababab"
local grammarFile = "/home/jonathan/Jonathan/programs/lua/wordmatch.bnf" --BNF file to use
local crossOp = "onepoint" --specifies which crossover operator to use
local selectionOp = "tournament_selection" --specifies the selection method
local crossoverProb = 0.7 --probability of crossover occuring
local mutationProb =0.01 --probability of mutation occuring on a codon
local tournamentSize = 3 -- size of tournament 
local truncSize = 0.3 --% of population that gets through evaluation
local eliteSize = 1 --no. of individuals that survive each evaluation intact
local maxGens = 100 --max number of Gens before reset
local popSize = 200 --how many individuals in your population
local maxChromeSize = 200
local maxWraps = 2
local defaultFitness = 10000
--GA Stuff
local initialised = false --checks if population has already been initialised
local generation = 0 --tracks no. of generations
local currentRun = 0 --how many times to run the experiment
local currentIndiv = 1 --tracks current individual
local individual ={fitness = defaultFitness, genome = {}, phenotype = "", usedCodons = 0}
local bestPop = {} --stores the best o the population
local population = {} --array storing individuals
local bestIndiv ={}
--grammar stuff
local non_terminals = {}
local terminals = {}
local rules = {}
local start_rule = ""
math.randomseed(os.time())


--utility methods
local function open_file(filename)
        local oldinput = io.input()
	io.input(filename)
	local file = io.input()
	io.input(oldinput)
	return file
     end

--tidies the strings(from PiL2 20.4, thanks PiL!)
function trim(s) 
   trimmedString  = s:gsub("^%s*(.-)%s*$", "%1")
  return trimmedString
end

local function print_individual(indiv)
   if indiv.usedCodons ==  nil then
      print("WTFFFFFfit: "..indiv.fitness.. " val: "..indiv.phenotype)
   else
      print("fit: "..indiv.fitness.. " val: "..indiv.phenotype.." used codons: "..indiv.usedCodons) 
   end
end

local function print_pop(pop)
   for i=1,table.getn(pop) do 
      print_individual(pop[i]) 
   end
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
      if ( math.mod(cur_input, table.getn(input)) == 0 and incremented) then
	 wraps = wraps +1
	 incremented =  false
      end

      current_symbol = table.remove(unexpanded_symbols)
      --add to output, if it's a terminal
      if current_symbol.type == T then
	 table.insert(output,current_symbol.value)
      else
	 local production_choices = rules[current_symbol.value]
	 local prod_idx = input[math.floor(math.mod(cur_input,table.getn(input))+1)]
	 prod_idx = math.floor(math.mod(prod_idx,table.getn(production_choices))+1)
	 
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
      return "",0
   else
      output = table.concat(output)
   end
   return output, cur_input
end

function create_individual(chromosome,fitVal)
   chromosome = chromosome or {}
   fitVal =  fitVal or defaultFitness
   if table.concat(chromosome) == "" then
      local chrome_size = math.random(10,maxChromeSize)
      for i = 1,chrome_size do
	 table.insert(chromosome, math.random(1,1000))
      end
   end
   local tmpVals,tmpCodons = map_chromosome(chromosome)
   local tmp_individual = {fitness = fitVal,genome = chromosome,phenotype = tmpVals, usedCodons = tmpCodons}
   return tmp_individual       
end

--methods for ranking population
local function get_fitness(phenotype)
   local fitness = math.max(string.len(phenotype),string.len(target))
   for i =1,math.min(string.len(phenotype),string.len(target)) do
      if(string.find(string.sub(phenotype,i,i),string.sub(target,i,i))) then
	 fitness = fitness -1
      end
   end   
   return fitness
end

local function sort_pop(pop,reverse)
   reverse = reverse or true
   if reverse  then
      table.sort(pop,function(a,b) return a.fitness < b.fitness end)
   else
      table.sort(pop,function(a,b) return a.fitness > b.fitness end)
   end
end

local function check_best(reverse)
   reverse =  reverse or true
   if reverse then
      if population[1].fitness < bestIndiv.fitness then
	 bestIndiv = create_individual(population[1].genome,population[1].fitness)
      end
   else
      if population[1].fitness > bestIndiv.fitness then
	 bestIndiv = create_individual(population[1].genome,population[1].fitness)
	 print("New best: ")
	 print_individual(bestIndiv)
      end
   end
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

function tournament_selection(pop)
   local newPop = {}
   while table.getn(newPop) < popSize do
      local competitors = {}
      for i = 1,tournamentSize do
	 local randIndiv =  math.random(1,popSize)
	 table.insert(competitors, pop[randIndiv])
      end
      sort_pop(competitors)
      table.insert(newPop,competitors[1])
   end
   return newPop
end   
   

function trunc_selection(pop)
   cutoff = math.floor(table.getn(pop)*truncSize)
   local newPop = {}
   for i=1,cutoff do
      tmp_individual = create_individual(pop[i].genome, pop[i].fitness)
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
    return newPop
end

--[[
creates a population of popsize, initialising 
 to 0 fitness and random phenotype
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
   initialised = true
end

local function run()
   init_pop()
   for i=1,maxGens do
      print("generation: "..i)
      while currentIndiv < popSize do
	 fitness = get_fitness(population[currentIndiv].phenotype)
	 population[currentIndiv].fitness = fitness
	 currentIndiv = currentIndiv + 1
      end
      currentIndiv = 1
      generation = generation +1
      sort_pop(population)
      check_best()
      bestPop = _G[selectionOp](population)
      population = generational_replace(bestPop)
      --mutating the population
      for i = eliteSize+1,table.getn(population) do
	 population[i] = intflip_mutation(population[i].genome)
      end
   end
   print_individual(bestIndiv)
end

run()
