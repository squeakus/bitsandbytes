local x = os.clock()
print(x)
local s = 0
for i=1,100000 do s = s + i 
   print(os.clock())
end
print(string.format("elapsed time: %.4f\n", os.clock() - x))