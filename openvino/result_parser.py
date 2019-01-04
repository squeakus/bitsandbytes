from tabulate import tabulate
from collections import OrderedDict
ncs1ov = []
ncs2ov = []
ncsdk1 = []
ncsdk2 = []
ncsdke = []
cpu = []

with open('results/ncs1_ovresult.txt', 'r') as results:
    for line in results:
        result = eval(line)
        ncs1ov.append(result)

with open('results/ncs2_ovresult.txt', 'r') as results:
    for line in results:
        result = eval(line)
        ncs2ov.append(result)

with open('results/ncs_ncsdk2result.txt', 'r') as results:
    for line in results:
        result = eval(line)
        ncsdk1.append(result)

with open('results/ncs2_ncsdk2result.txt', 'r') as results:
    for line in results:
        result = eval(line)
        ncsdk2.append(result)

with open('results/nce_ncsdk2result.txt', 'r') as results:
    for line in results:
        result = eval(line)
        ncsdke.append(result)

with open('results/i7_8700_3.2ghz_ovresult.txt', 'r') as results:
    for line in results:
        result = eval(line)
        cpu.append(result)

networks = {}
for result in ncs1ov:
    key = result['network']
    networks[key] = [round(result['inftime'], 2)]
for result in ncs2ov:
    key = result['network']
    networks[key].append(round(result['inftime'], 2))
for result in cpu:
    key = result['network']
    networks[key].append(round(result['inftime'], 2))
for result in ncsdk1:
    key = result['network']
    if result['inftime'] is not "NA":
        networks[key].append(round(result['inftime'], 2))
    else:
        networks[key].append("NA")
for result in ncsdk2:
    key = result['network']
    if result['inftime'] is not "NA":
        networks[key].append(round(result['inftime'], 2))
    else:
        networks[key].append("NA")
for result in ncsdke:
    key = result['network']
    if result['inftime'] is not "NA":
        networks[key].append(round(result['inftime'], 2))
    else:
        networks[key].append("NA")

networks = OrderedDict(sorted(networks.items()))
net_results = []

for key in networks:
    ncs1ov = networks[key][0]
    ncs2ov = networks[key][1]
    cpu = networks[key][2]
    ncsdk1 = networks[key][3]
    ncsdk2 = networks[key][4]
    ncsdke = networks[key][5]

    ncs1ovfps = round(1000 / ncs1ov, 1)
    ncs2ovfps = round(1000 / ncs2ov, 1)
    if ncsdk1 is not "NA":
        ncsdk1fps = round(1000 / ncsdk1, 1)
    else:
        ncsdk1fps = "NA"
    if ncsdk1 is not "NA":
        ncsdkefps = round(1000 / ncsdke, 1)
    else:
        ncsdkefps = "NA"

    cpufps = round(1000 / cpu, 1)


    # ncs2ovspeedup = round(ncs1ov/ncs2ov, 1)
    # cpuspeedup = round(ncs1ov/cpu, 1)
    speedup = round(ncs2ovfps/ncs1ovfps,1)
    result = [key, ncs1ovfps, ncs2ovfps, speedup]
    net_results.append(result)

headers = ["Network", "NCS1", "NCS2", "Speedup"]
print()
print(tabulate(net_results, headers=headers, tablefmt="latex"))
