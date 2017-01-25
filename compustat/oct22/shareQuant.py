import xlrd, csv

wb = xlrd.open_workbook('shareQuantity.xlsx')
sh = wb.sheet_by_index(0)

csvWriter = csv.writer(open('shareQuant.csv', 'wb'), 
                       delimiter=',', 
                       quoting=csv.QUOTE_NONE)

for rownum in range(1,sh.nrows):
    row_vals = sh.row_values(rownum)
    row_vals.pop(1) # empty col
    csvWriter.writerow(row_vals)
