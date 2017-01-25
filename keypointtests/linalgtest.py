from numpy import linalg

descriptors = [[10,0,0,0,0,0,0,0], [5,100,12,8,18,4,1,1,1,20]]

for idx,_ in enumerate(descriptors):
    print "before:", descriptors[idx]
    descriptors[idx] =  descriptors[idx] / linalg.norm(descriptors[idx])
    #descriptors[point] = descriptors[point] / linalg.norm(descriptors[point])
    print "normed:", descriptors[idx]
    for idy, elem in enumerate(descriptors[idx]):
        if elem > 0.2:
            descriptors[idx][idy] = 0.2
    print "trunc:", descriptors[idx]
    
    descriptors[idx] =  descriptors[idx] / linalg.norm(descriptors[idx])
    print "normed again:", descriptors[idx]
    descriptors[idx] =  descriptors[idx] * 512
    print "scaled to 512:", descriptors[idx].astype(int)
