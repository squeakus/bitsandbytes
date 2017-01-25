local gram_file = "example2.bnf"
local NT = "NT" --flags denoting whether a symbol is terminal or non-terminal
local T = "T"
local non_terminals = {}
local terminals = {}
local rules = {}
local non_term_pattern = "[<>]"
local rule_separator = "::="
local separator = "|"
local start_rule = ""
local individual ={}
math.randomseed(os.time())

function createIndividual()
   local chrome_size = math.random(10,100)
   for i = 1,chrome_size do
      individual[i] = math.random(1,1000)
   end
end

function printRules()
   print("no of rules: "..table.getn(rules))
   for lhs, prods in pairs(rules) do
      print("Rule: "..lhs)
      for i = 1,table.getn(prods) do 
	 for index, production in pairs(prods[i]) do
	    print(index, production.value, production.type)
	 end
	 print("-----------")
      end
      print("---------")
   end
end

function trim(s)
  -- from PiL2 20.4
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

local function open_file(filename)
   local oldinput = io.input()
   io.input(filename)
   local file = io.input()
   io.input(oldinput)
   return file
end

local function read_bnf_file()
   local tmpFile = open_file(gram_file)
   print("reading file")
   while true do
      local line = tmpFile:read('*l')
      --initialising the production list for each rule
      local production_table = {}
      if not line then break end
      
      -- check for comments and empty lines
      if not(string.sub(line,1,1) == '#' or trim(line) == "") then
	 -- if its a rule, break it into left and right parts
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
		  production = string.gsub(production,"<%a*>"," %1 ")
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

local function generate(input, maxwraps)
   local cur_input = 0
   local current_symbol = {}
   local incremented = false
   local wraps = 0
   local output = {}
   local loopCounter =0
   local unexpanded_symbols = {start_rule}
   
   while wraps < maxwraps and table.getn(unexpanded_symbols) >0 do
      --expand wraps
      if math.mod(cur_input,table.getn(input)) == 0 and incremented then
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
      return 0, cur_input
   end
   output = table.concat(output," ")
   return output, cur_input
end

read_bnf_file()
--printRules()
createIndividual()
local result, codons = generate(individual,3)
print("result: "..result.." codons used: "..codons)
--add_hook("new_game","start",read_bnf_file)
