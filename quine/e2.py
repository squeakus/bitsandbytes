import random
x = ['0',
     'dna=[random.randint(0,1)for _ in range(5)]',
     'result=sum(dna)',
     'print result',
     'x[0]=str(result)',
     'if result == 5: print "found it!",x[0]; exit()',
     'outfile = open("out.py","w")',
     'outfile.write("import random"+"\\n")',
     'outfile.write("x =" + str(x)+"\\n")',
     'outfile.write("y =" + str(y)+"\\n")']
y = ['for m in x:exec(m)',
     'for m in y:outfile.write(m+"\\n")',
     ]

for m in x: exec(m)
for m in y: outfile.write(m+"\n")
    
