# the neural net part
import pylab, numpy
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import CrossValidator, ModuleValidator

results = pylab.loadtxt('credit.txt')
target = results[:,-1]
data = numpy.delete(results,-1,1)

#print "data", tuple(data[0])
#print "target", (target[0],)

#net = buildNetwork(14, 10, 1)
net = buildNetwork(14, 10, 1, hiddenclass=TanhLayer)
#print net.activate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

ds = SupervisedDataSet(14, 1)

for i in range(len(data)):
    ds.addSample(tuple(data[i]), (target[i],))

trainer = BackpropTrainer(net, ds)
evaluation = ModuleValidator()
validator = CrossValidator(trainer=trainer, dataset=trainer.ds, n_folds=5, valfunc=evaluation.MSE)
print(validator.validate())
