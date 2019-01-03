from tabulate import tabulate

ncs1 = []
ncs2 = []
ncsdk1 = []
ncsdk2 = []
ncsdke = []
cpu = []

with open('results/ncs1_ovresult.txt', 'r') as results:
    for line in results:
        result = eval(line)
        ncs1.append(result)

with open('results/ncs2_ovresult.txt', 'r') as results:
    for line in results:
        result = eval(line)
        ncs2.append(result)

#with open('results/ncs_ncsdk2result.txt', 'r') as results:
#    for line in results:
#        result = eval(line)
#        ncsdk1.append(result)

#with open('results/ncs2_ncsdk2result.txt', 'r') as results:
#    for line in results:
#        result = eval(line)
#        ncsdk2.append(result)

#with open('results/nce_encsdk2result.txt', 'r') as results:
#    for line in results:
#        result = eval(line)
#        ncsdke.append(result)

with open('results/i7_8700_3.2ghz_ovresult.txt', 'r') as results:
    for line in results:
        result = eval(line)
        cpu.append(result)


networks = {}
for result in ncs1:
    key = result['network']
    networks[key] = [round(result['inftime'],2)]
for result in ncs2:
    key = result['network']
    networks[key].append(round(result['inftime'],2))
for result in cpu:
    key = result['network']
    networks[key].append(round(result['inftime'],2))

net_results = []
for key in networks:
    ncs1 = networks[key][0]
    ncs2 = networks[key][1]
    cpu = networks[key][2]
    ncs1fps = round(1000 / ncs1, 0)
    ncs2fps = round(1000 / ncs2, 0)
    cpufps = round(1000 / cpu, 0)
    ncs2speedup = round(ncs1/ncs2, 1)
    cpuspeedup = round(ncs1/cpu, 1)

    result = [key, ncs1, ncs2, cpu, ncs1fps, ncs2fps, cpufps, ncs2speedup, cpuspeedup]
    net_results .append(result)

headers = ["network","ncs1", "ncs2", "cpu","ncs fps", "ncs2 fps",
           "cpu fps", "ncs2 speedup", "cpu speedup"]
print(tabulate(net_results, headers=headers))
