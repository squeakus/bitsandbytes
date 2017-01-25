import math

target = 600851475143
targSqRoot = math.sqrt(target)
factor = 1
found = False

def checkPrime(number):
     for j in range(2,number):
          if number % j == 0:
               return False
     return True

# while found == False:
#      if target % factor ==0:
#           print("factor: ",factor)
#           found = checkPrime(factor)
#           if found == True:
#                print(factor)
#                found=False
#      factor = factor +1

checkPrime(59569)

