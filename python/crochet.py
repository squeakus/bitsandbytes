
results_file = open('results.dat', 'w')
row = 20
double_stitch = 5

for i in range(100):
    new_row = 0
    for j in range(row):        
        if j % double_stitch == 0:
            new_row += 2
        else:
            new_row += 1
    print "row", i, ":", new_row
    results_file.write(str(i) + ' ' + str(new_row) + '\n')
    row = new_row
results_file.close()
