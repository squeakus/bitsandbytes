individual ={}
population = {}
popSize = 10
eliteSize = 10
bestIndiv = {fitness=0,jointVal=0}
math.randomseed( os.time() )


local function printPop()
   for i=1,table.getn(population) do 
      print("fitness "..population[i].fitness,"value: "..table.concat(population[i].jointVals)) end
   end

local function sortPop()--awful bubblesort,sort it out
--   for i=1,eliteSize do
--      for i=table.getn(population)-1,1,-1 do --start from end
--	 if population[i].fitness < population[i+1].fitness then
--	    table.insert(population,i,population[i+1])
--	    table.remove(population,i+2)
--	 end
--      end
--   end
    table.sort(population,function(a,b) return a.fitness > b.fitness end)

end

local function checkBest()
   if population[1].fitness > bestIndiv.fitness then
      bestIndiv = population[1]
      bestFile = io.open("best.txt","a")
      bestFile:write(bestIndiv.fitness..": "..table.concat(bestIndiv.jointVals,",").."\n")
   end
end


--this will be the initialising function
for i=1,popSize do 
   valArray ={}
   for i=1,14 do
      table.insert(valArray,math.random(1,4))
   end
   individual = {fitness = math.random(100),jointVals = valArray}
   table.insert(population,individual)
end


printPop()
--checkBest()
sortPop()
print("Sorted pop")
printPop()
--checkBest()

