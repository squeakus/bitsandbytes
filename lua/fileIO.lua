module(...,package.seeall);

local function OpenFile(filename)
	local oldinput = io.input()
	io.input(filename)
	local file = io.input()
	io.input(oldinput)
	return file
     end

local function writeFile(filename)
	local oldoutput = io.output()
	io.output(filename)
	local file = io.output()
	io.output(oldoutput)
	return file
     end

local function test_read()
   local file = OpenFile("best.txt")
   echo(file:read("*l"))
   file:close()
   --local file2 = writeFile("moo.txt")
   --file2:write("hey")
   --file2:close()
end
