import analyser, subprocess, graph 
from geometry import *

loadFile = open("./test.dat",'r')
for line in loadFile:
  if line.startswith("phenotype:"):
    line = line.lstrip("phenotype:")
    phenotype = line
print "writing program to test.py"

footer = open("./footer.txt",'r')
saveFile = open("./test.py",'w')
imports= "import analyser, subprocess, graph\nfrom geometry import *\n"
saveFile.write(imports)
saveFile.write(analyser.python_filter(line)+"\n")
for line in footer:
  saveFile.write(line)
saveFile.close

analyser = analyser.Analyser('test',phenotype,True)
analyser.create_graph()
#analyser.parse_graph(analyser.myGraph)




#using medit to show the graph
meshName = "indiv.test.mesh"
cmd = "ffmedit "+meshName
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
process.communicate()

#using slffea to show the mesh
analyser.apply_stresses()
analyser.create_slf_file()
analyser.test_slf_file()
analyser.parse_results()
analyser.print_stresses()
analyser.show_analysis()

#using matplot to show the graph
#analyser.myGraph.show_picture()
