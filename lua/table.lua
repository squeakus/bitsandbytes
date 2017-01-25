local testTable = {5,6,7,8,9}
table.insert(testTable,5)
print(table.concat(testTable,","))

--Insertion overwrites the value
--testTable[1] = 6
--print(table.concat(testTable,","))

--table.insert(testTable,1,3)
--for index, value in pairs(testTable) do print(index,value) end
--table.insert(testTable,1,4)
--table.insert(testTable,1,2) 

--print("2 then 4")
--for index, value in pairs(testTable) do print(index,value) end

--table.insert(testTable,0,7) 
--print("table starts at 1, so 0 it sticks it on at the end")
--for index, value in pairs(testTable) do print(index,value) end

--testTable:insert(1,5)
--print("syntacic sugar doesnt work")
--for index, value in pairs(testTable) do print(index,value) end
