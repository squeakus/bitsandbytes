import subprocess

cmd = "java -jar ISSVis.jar -csv testb70y0.csv"

process = subprocess.Popen(cmd, shell=True,
                           stdout=subprocess.PIPE,
                           stdin=subprocess.PIPE)

# extract wattage from the result
result = process.communicate()
result = result[0].split('\n')
found = False
for line in result:
    if line.startswith("Score"):
        line = line.split('=')
        fitness = float(line[1])
        found = True
if found == False:
    errfile = open("err.log", 'a')
    errfile.write(result)
    errfile.close()
    fitness = 100
