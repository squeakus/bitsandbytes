import math, random

def ave(values):
    return float(sum(values)) / len(values)

def std(values, ave):
    return math.sqrt(float(sum((value - ave) ** 2
                               for value in values)) / len(values))


def compare_averages(gcount):
    groups = []
    total = []
    
    for i in range(gcount):
        run = []
        for j in range(10):
            x = random.randint(0, 100)
            run.append(x)
            total.append(x)
        groups.append(run)

    ave_list = []
    std_list = []
    
    for group in groups:
        average = ave(group)
        std_dev = std(group, average)
        ave_list.append(average)
        std_list.append(std_dev)

    t_average = ave(total)
    t_std_dev = std(total, t_average)
    print "t-ave:",t_average, "t-std:", t_std_dev
    print "g-ave:",ave(ave_list), "g-std:", ave(std_list)
    #but what if we get the avr of the std 
    #test_ave = ave(std_list)
    #print "test", std(std_list, test_ave)
    
for _ in range(10):
    compare_averages(100)
