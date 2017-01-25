import random, subprocess, sys
x = ['0',
     'out = open("out.py","w")',
     'dna=[random.randint(0,1)for _ in range(5)]',
     'result=sum(dna)',
     'x[0]=str(result)',
     'print x[0]',
     'if x[0] == "5":out.write("found it! ans="+x[0]+str(dna)); exit()',
     'out.write("import random, subprocess, sys\\n")',
     'out.write("x =" + str(x)+"\\n")',
     'out.write("y =" + str(y)+"\\n")']
y = ['for m in x:exec(m)',
     'for m in y:out.write(m+"\\n")',
     'subprocess.Popen(["/usr/bin/python", "out.py"])',
     'sys.exit()']


for m in x: exec(m)
for m in y: out.write(m+"\n")
    
