import math

target = 600851475143
targSqRoot = 1000000
finished = False
targetList = range(2,int(math.floor(targSqRoot)))
index = 0;
while not finished:
    num = targetList[index]
    print("evaluating num: ",num)
    listSize = len(targetList)
    counter = index+1
    while counter < listSize:
        if(targetList[counter]%num == 0):
            del(targetList[counter])
            listSize = listSize -1
        counter =  counter + 1
    index = index +1
    if index >= len(targetList):
        finished = True

FILE = open("primes.txt","w")
for element in targetList:
    FILE.writelines((str(element)+'\n'))
FILE.close()

