import random
import numpy as np

def init_weights(net,prob):
    for node in np.nditer(net, op_flags=['readwrite']):
        if random.random() < prob:
            node[...] = (random.random() * 2) - 1
    
def main():
    size = (3,2) 
    mask = random(size) > 0.01

    #hidden = (random(resSize,resSize)*2)-1
    #hidden = np.zeros([5,5])
    #out = np.zeros([5])
    #init_weights(hidden,1)
    #init_weights(out,1)
    
    #print hidden
    
    #rhoW = max(abs(np.linalg.eig(hidden)[0]))
    #print np.linalg.eig(hidden)[0]
    #print "the spectral radius is",rhoW
    
if __name__ == "__main__":
    main()
