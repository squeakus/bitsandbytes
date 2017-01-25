-- tutorialscript.lua
-- A script that will be used in Jok's tutorial

-- The two dashes are what signal Lua not to use this line. Called "commenting".

--[[ This type of commenting stretches
across multiple
lines
like this ]]--


---------

local function start() -- this is the main function. int main() { } in C/C++, begin in Pascal, etc. 
--This is what gets run as soon as the script starts

helloworld() -- this is called a function call. It basically means it'll go to the other FUNCTION helloworld, then execute it

end -- This ends every function and statement (if then else, for - do, while - do, etc)

---------

local function helloworld() -- And this is the function that gets called. Looks almost the same as start()

local a = 1337 -- giving a variable (a) a value (1337) is called assigning, 
		  -- and is done by simply writing var = value, replacing the two with their actual values
local b = "Who is" -- almost the same as the first variable, this var, b, is a string. Basically a row of characters.
local c = "Jok is!" -- same thing here. If you print them, to make a new line, use the \n operand, writing, for example "Jok is!\n"

local d = b .. " " .. a -- This is called concatenating a string. It merges the two values, and gives out a string of the two.
				 -- In this case "Who is 1337"
echo(d) -- The echo() command writes a message to the chat area, same as if you were doing /echo in-game.

centerprint("",c,"",100) -- The centerprint() command writes three lines of text, in the middle of the screen.
				-- the first argument ("") is the first line, second (the string c) is the middle, third is the last
				-- The fourth argument (100) is the time the text will be displayed on the screen

end


----------


--Now for the tough bit. A hook is a 'function' that instead of having to call it when you use it, will be attached to the end of the game's default functions
--For example, attaching a hook to a function, the hook being "new_game", the function will be ran at every new game, with no other necessities

local function Hello()
local a = get_player_info(0) -- This function will return a table of two things, name and score. These are accessed with a.score, a.name, etc 
local b = get_player_info(1) -- Read the readme for the full list of commands

draw_text("Welcome ".. a.name ..". Good luck to ".. b.name .." as well.", 0, 300) -- this function draw text (the strings) to a specific spot on the game screen
													  -- x = 0, and y = 300. Measured in pixels

end

add_hook("new_game","unique_group_name",Hello) 

