import pylab
rank1 = [[1,1],[1,3],[3,1]]
rank2 = [[2,3],[5,5],[3,3]]

population = [rank1, rank2]
numcolors = len(population)
cm = pylab.get_cmap('gist_rainbow')

for idx, front in enumerate(population):
    color = cm(1.*idx/len(population))
    for indiv in front:
        pylab.scatter(indiv[0], indiv[1], s=100, c=color, alpha=0.7)

pylab.grid(True)
pylab.title("Pareto Optimisation")
pylab.xlabel("Lift Maximisation")
pylab.ylabel("Drag Minimisation")
pylab.show()

# all the colors of the rainbow
# 
# for i in range(NUM_COLORS):
#     color = cm(1.*i/NUM_COLORS)
