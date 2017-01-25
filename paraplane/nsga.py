""" Methods used for NSGA multiobjective fitness selection"""

def fast_nondominated_sort(pop):
    """nsga2 algorithm creates a list of fronts"""
    fronts = []
    dom_list = dict()
    for A in pop:
        dom_list[hash(A)] = []
        A.domCount = 0
        
        for B in pop:
            if A.dominates(B):
                dom_list[hash(A)].append(B)
            elif B.dominates(A):
                A.domCount += 1

        if A.domCount == 0:
            A.rank = 0
            fronts.append([]) # add new front
            fronts[0].append(A) # add to first front

    i = 0 #front counter
    while len(fronts[i]) != 0:
        new_pop = []
        
        for A in fronts[i]:
            for B in dom_list[hash(A)]:
                B.domCount -= 1
                
                if B.domCount == 0:
                    B.rank = i + 1

                    new_pop.append(B)
        i += 1
        fronts.append([])
        fronts[i] = new_pop
    count_fronts(fronts)
    return fronts

def crowded_comparasion_operator(x, y):
    if x.rank < y.rank:
        return 1
    elif (x.rank == y.rank):
        if x.distance > y.distance:
            return -1
        else:
            return 1

def crowding_distance_assignment(pop):
    """calulates distances between individuals"""
    pop_size = len(pop)
    for indiv in pop:
        indiv.distance = 0
    #assign cumulative distance to each individual
    for val in range(len(pop[0].fitness)):
        pop.sort(lambda x, y: cmp(x.fitness[val], y.fitness[val]))
        #always include boundary points
        pop[0].distance = 9999
        pop[pop_size-1].distance = 9999
        for i in range(2, pop_size-1):
            pop[i].distance += (pop[i + 1].fitness[val]
                                - pop[i - 1].fitness[val])
        #send back a sorted pop
    pop.sort(crowded_comparasion_operator)

def count_fronts(fronts):
    front_cnt = 0
    for front in fronts:
        if len(front) > 0:
            front_cnt += 1
    print str(front_cnt), "valid fronts out of", len(fronts)
    return front_cnt
