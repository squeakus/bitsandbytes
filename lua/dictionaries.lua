players ={}
player = {}
player["name"] = "Haywood Slap"
player["score"] = 0
player["kills"] = 0
player["deaths"] = 0

-- We can also use variables that contain strings for either 
-- the key or value:
key = "name"
value = "Haywood Slap"
player[key] = value

-- Lua also provides shortcuts for setting table values 
-- that use strings as the key.
player.name = "Haywood Slap"

-- is shorthand for
player["name"] = "Haywood Slap"

-- Lua also has a shorthand method that allows us to initialize all the
-- fields at one.  This is equivalent to the first method.
player2 = {
    name = "woodhay Slap",
    score = 0,
    kills = 0,
    points = 0
}
print(player.name)
print(player2.name)
table.insert(players,player)
table.insert(players,player2)
for i=1,table.getn(players) do print(players[i].name) end



