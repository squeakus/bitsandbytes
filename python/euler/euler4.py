done = False
palindrome = False

def checkPalindrome(a,b):
    num = a*b
    numString = str(num)    
    numList = []
    for char in numString:
        numList.append(char)
    for i in range(0,len(numList)/2):
        if numList[i] != numList[len(numList)-(i+1)]:
            return False
    print("iVal: ",a,"jVal ",b)
    print(num)
    return True

for i in range(994,100,-1):
    if done:
        break
    for j in range(999,100,-1):
        palindrome = checkPalindrome(i,j)
        if palindrome:
            done = True
            break
