import random
x = ['dna=[random.randint(0,1)for _ in range(5)]',
     'result=sum(dna)',
     'x[3]=str(result)',
     '0', 
     'print "x =", x',
     'print "y =", y']
y = ['import random','for m in x:exec(m)','for m in y:print m']

for m in x: exec(m)
for m in y: print m
    
