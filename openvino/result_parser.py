from tabulate import tabulate
from collections import OrderedDict

def main():
    experiments = {}
    net_names = set()
    experiments = load_result('ncs1ov', 'results/ncs1_ovresult.txt', experiments)
    experiments = load_result('ncs1ovusb2', 'usb2results/ncs1_ovresult.txt', experiments)
    experiments = OrderedDict(sorted(experiments.items()))
    explist = ['ncs1ovusb2', 'ncs1ov', 'speedup']
    headers = ["Network", "NCS1 USB2", "NCS1 USB3", 'speedup']

    table_results(experiments, explist, headers)


    # for key in networks:
    #     ncs1ov = networks[key][0]
    #     ncs2ov = networks[key][1]
    #     cpu = networks[key][2]
    #     ncsdk1 = networks[key][3]
    #     ncsdk2 = networks[key][4]
    #     ncsdke = networks[key][5]

    #     ncs1ovfps = round(1000 / ncs1ov, 1)
    #     ncs2ovfps = round(1000 / ncs2ov, 1)
    #     if ncsdk1 is not "NA":
    #         ncsdk1fps = round(1000 / ncsdk1, 1)
    #     else:
    #         ncsdk1fps = "NA"
    #     if ncsdk1 is not "NA":
    #         ncsdkefps = round(1000 / ncsdke, 1)
    #     else:
    #         ncsdkefps = "NA"

    #     cpufps = round(1000 / cpu, 1)


    #     # ncs2ovspeedup = round(ncs1ov/ncs2ov, 1)
    #     # cpuspeedup = round(ncs1ov/cpu, 1)
    #     speedup = round(ncs2ovfps/ncs1ovfps,1)
    #     result = [key, ncs1ovfps, ncs2ovfps, speedup]
    #     net_results.append(result)

    # headers = ["Network", "NCS1", "NCS2", "Speedup"]
    # print()
    # print(tabulate(net_results, headers=headers, tablefmt="latex"))

def table_results(experiments, explist, headers):
    results = []
    for key in experiments:
        print(key)
        result = []
        result.append(key)
        for exp in explist:
            if exp in experiments[key]:
                inftime = experiments[key][exp]
                fps = round(1000 / inftime, 1)
                result.append(fps)
            elif exp == 'speedup' and 'NA' not in result:
                result.append(round(result[-1]/result[-2],1))
            else:
                result.append("NA")

        results.append(result)
    # print(tabulate(results, headers=headers, tablefmt="latex"))
    print(tabulate(results, headers=headers))

def load_result(expname, filename, experiments):
    with open(filename, 'r') as resultsfile:
        for line in resultsfile:
            result = eval(line)
            network = result['network']
            inftime = result['inftime']
            if network in experiments:
                experiments[network][expname] = inftime
            else:
                experiments[network] = {expname: inftime}

    return experiments


if __name__ == '__main__':
    main()