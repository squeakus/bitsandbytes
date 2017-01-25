num = 2520
found = False

while not found:
    found = True
    for i in range(11,20):
        if num%i !=0:
            found = False
            num = num +1
            break
    if found == True:
        print num 
