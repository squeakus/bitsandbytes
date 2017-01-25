local moves = 3 
local moveLoaded = false
local initialised = false
local lastFrame = 0
local currentFrame =0             --counter to track frames
local moveFrame = 1000            --make a move after a certain amount of turns
local jointIndex =0
local jointVals = {}
local currentIndiv =1
local popFile = "best.txt"
local popSize = 0
local population = {}


local function print_pop(pop)
   for i=1,table.getn(pop) do 
      print("fitness "..pop[i].fitness,"value: "..table.concat(pop[i].jointVals,",")) 
   end
end

local function get_fitness()
   tori_score = math.floor(get_player_info(1).injury)
   uke_score = math.floor(get_player_info(0).injury)
   fitness = tori_score - uke_score
   return fitness
end

local function open_file(filename)
	local oldinput = io.input()
	io.input(filename)
	local file = io.input()
	io.input(oldinput)
	return file
     end

local function load_pop(fileName)      
   population ={}
   tmpFile = open_file(fileName)

   while true do
      jVals ={}
      fitVal = 0
      indivString = tmpFile:read("*l")
      if not indivString then break end
      jointStart = string.find(indivString,":")+1
      jointString = string.sub(indivString,jointStart)        
      fitVal =string.sub(indivString,0,jointStart)
      for val in jointString:gmatch("%w+") do
	 table.insert(jVals,val)
      end
      indiv ={fitness= fitVal, jointVals = jVals}
      table.insert(population,indiv)
   end
   print("pop loaded from file")
   initialised = true
   print_pop(population)
   tmpFile:close()
   popSize = table.getn(population)
   print("popsize: "..popSize)
   echo("popsize: "..popSize)
   print("press n for next move, p from previous,enjoy!")
   echo("press n for next move, p from previous,enjoy!")
end

-- takes an array of joint values and configures tori
local function make_move(valArray)
   print("Making move")
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
   if moveLoaded == false then      
      print("loaded move:"..currentIndiv.." of "..popSize)
      echo("loaded move:"..currentIndiv.." of "..popSize)
      move = population[currentIndiv].jointVals
      print("move: "..table.concat(move,","))
      print("recorded fitness: "..population[currentIndiv].fitness)
      moveLoaded = true
   end
   if currentFrame == 0 then
      make_move(move)
   end
   currentFrame = (currentFrame+1) % moveFrame
   step_game()
end

function keyDown(key)
   if(key == string.byte("n")) then
      currentIndiv = (currentIndiv+1) % (popSize+1)
      echo("next move is "..currentIndiv.." of "..popSize)
   end
   if(key == string.byte("p")) then
      currentIndiv = (currentIndiv-1) % (popSize+1)
      echo("previous move is "..currentIndiv.." of "..popSize)
   end
   if currentIndiv == 0 then
      print("making sure it doesn't trip off the end of the array")
      currentIndiv =1
   end
end

local function start()
   if initialised == false then
      load_pop(popFile)
      initialised = true
   end
   rules =get_game_rules()
   lastFrame = rules.matchframes - 50
   moveFrame = math.floor((lastFrame / rules.turnframes) /(moves-1))
   next_turn()
end

local function next_game()
   moveLoaded = false
   fitness = get_fitness()
   echo("fitness: "..fitness)
   print("actual fitness: "..fitness)
   jointIndex =0
   currentFrame =0
   start_new_game()
end

--add_hook("enter_freeze","keep stepping",next_turn)
add_hook("enter_freeze","keep stepping",next_turn)
add_hook("end_game","start next game",next_game)
add_hook("new_game","start",start)
add_hook("key_down","keys",keyDown)
