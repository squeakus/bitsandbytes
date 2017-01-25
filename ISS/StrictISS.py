import random, re, cma, subprocess
SARJvel = 0.15 #deg/sec 
SARJacc = 0.005 #deg/sec^2
BGAvel = 0.25
BGAacc= 0.01

#optimised starting positions
beta70best = [205.079297538498, 178.26765893171793, 341.24720919821618, 200.00730753557255, 20.45388927122956, 160.80737856381572, 0.0, 159.56381666495884, 342.48953832046874, 200.50991770402038]

betaminus72best = [181.66199135482427, 347.93812704684564, 196.56821721193921, 11.666796952001137, 166.20113352252014, 339.52231368066907, 162.97448689036696, 342.74985271675433, 196.13158148121408, 360.0]



def genome2accel(genome):
    #create chunks for each timestep: 92 x [10]
    sliced = []
    for i in range(0,920,10):
        sliced.append(genome[i:i+10])

    # scale accelerations to limits:
    scaled = []
    scaled_vec = [0]*10
    for vec in sliced:
        for i in range(len(vec)):
            if i < 3:
                scaled_vec[i] = (vec[i] * (2 * SARJacc)) - SARJacc
            else:
                scaled_vec[i] = (vec[i] * (2 * BGAacc)) - BGAacc
        scaled.append(scaled_vec)
    return scaled
    
def accel2vel(accel_array):
    vel_array = [accel_array[0]]

    for i in range(1,len(accel_array)):
        accel_vec = accel_array[i]
        new_vel = []
        for j in range(len(accel_vec)):
            #limit dependent on vector position
            if j < 3:
                vel_limit = SARJvel
            else:
                vel_limit = BGAvel
            speed = accel_vec[j] + vel_array[i-1][j]
            #only increment speed if below limit
            if speed < vel_limit and speed > -vel_limit:
                new_vel.append(speed)
            else:
                new_vel.append(vel_array[i-1][j])
        vel_array.append(new_vel)
    return vel_array    

def checkcyclic(vel_array):
    #star op unpacks array into positional args
    deltas = map(sum, zip(*vel_array))
    return deltas
    
def vel2pos(init_pos, vel_array):
    pos_array = [init_pos]
    # NOT GOOD! only need 91 velocities
    for i in range(len(vel_array)-1):
        new_pos = []
        for j in range(len(vel_array[i])):
            pos = ((vel_array[i][j]*60) + pos_array[i][j]) % 360
            new_pos.append(pos)
        pos_array.append(new_pos)
    return pos_array
    
def write_csv(name, configstr, pos_array, vel_array):
    outfile = open(name, 'w')
    outfile.write("#header\n")    
    #generate line for each minute
    for i in range(92):
        resultstr = configstr
        for j in range(len(pos_array[i])):
            resultstr += ","+str(pos_array[i][j])+","+str(vel_array[i][j])
        
        resultstr += '\n'
        outfile.write(resultstr)
    outfile.close()

def eval_config(name, beta, yaw, init_pos, genome, render=False):
    """
    1: read the genome into acceleration chunks
    2: generate a vel_array from the accel_array
    3: check if the vel array is cyclic
    4: generate the pos_array from the vel_array
    Create a csv and then run simulator, extract power ratio
    csv format:
    beta, yaw, SSARJ angle, SSARJ speed, PSARJ angle, PSARJ speed, 
    BGA 1A angle, BGA 1A speed, BGA 2A angle, BGA 2A speed......
    """
    name = name+"b"+str(beta)+"y"+str(yaw)+".csv"
    configstr = str(beta) + "," + str(yaw)
    accel_array = genome2accel(genome)
    vel_array = accel2vel(accel_array)
    deltas = checkcyclic(vel_array)
    deltotal = 0
    for delta in deltas:
        deltotal += abs(delta)
    
    pos_array = vel2pos(init_pos, vel_array)
    write_csv(name, configstr, pos_array, vel_array)
    cmd = "java -jar ISS.jar -csv "+ name
    if render:
        cmd = cmd + " -render"
        
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)

    # extract fitness from the result
    resultout = process.communicate()
    result = resultout[0].split('\n')
    found = False
    for line in result:
        if line.startswith("Score"):
            line = line.split('=')
            fitness = float(line[1])
            found = True
    if found == False:
        errfile = open("err.log", 'a')
        errfile.write(str(resultout[0]))
        errfile.close()
        fitness = 100.0

    deltapunished = fitness / (deltotal / 10)
    return deltapunished

def main():

    optim = cma.CMAEvolutionStrategy([0.5]*920, 0.1, {'bounds': [0, 1]})

    for i in range(150):
        print "generation", i
        solutions = optim.ask()
        fitnesses = []
        moo = 0
        for soln in solutions:
            fitness = eval_config("test", 70, 0, beta70best, soln)
            minimize = 100000000 - fitness
            print fitness
            fitnesses.append(minimize)

        optim.tell(solutions, fitnesses)

        #write results
        bestgenome = list(optim.result()[0])
        bestfit = 100000000 - optim.result()[1]
        print "best", bestfit, optim.result()[1]
        beststr = str(bestfit) + " : " + str(bestgenome) + "\n"

        resfile = open("cmaresult.dat",'a')
        resfile.write(beststr)
        resfile.close()

if __name__ == "__main__":
    bestfile = open('cmaresult.dat','r')
    for line in bestfile:
        bestresult = line
    bestgenome = eval(bestresult.split(':')[1])
    print "best", bestgenome
    eval_config("test", 70, 0, beta70best, bestgenome, True)
    #main()
