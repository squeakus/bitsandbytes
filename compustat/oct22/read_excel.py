import xlrd, csv, graph
import matplotlib.mlab as mlab
import numpy as np


def interpolate_points(point_a, point_b, n_points):
    new_list = [point_a]
    line_size = (float(point_b[0]-point_a[0]), float(point_b[1]-point_a[1]))
    x_step = line_size[0] / (n_points + 1)
    y_step = line_size[1] / (n_points + 1)
    for i in range(1, n_points+1):
        point = (point_a[0]+(x_step * i),point_a[1]+(y_step * i))
        new_list.append(point)
        
    new_list.append(point_b)
    return new_list

def interpolate(a, b, n_points):
    new_list = []
    difference = float(b) - float(a)
    x_step = difference / (n_points + 1)
    for i in range(1, n_points+1):
        point = float(a)+(x_step * i)
        new_list.append(point)
        
    new_list.append(b)
    return new_list

def exceltocsv(spreadsheet, csvname):
    wb = xlrd.open_workbook(spreadsheet)
    share_wb = xlrd.open_workbook('shareQuantity.xlsx')
    
    sh = wb.sheet_by_index(0)
    share_sheet = share_wb.sheet_by_index(0)
    
    csvWriter = csv.writer(open(csvname, 'wb'), 
                           delimiter=',', 
                           quoting=csv.QUOTE_NONE)

    for rownum in range(1, sh.nrows):
        row_vals = sh.row_values(rownum)
        row_vals.pop(1) # empty col
        row_vals.pop(-1) # faulty beta
        if not any(item in ('@NA', '@NM', '@CF', '@SF', '@AF')
                   for item in row_vals):

            #add in share values
            for sharenum in range(0,share_sheet.nrows):
                share_vals = share_sheet.row_values(sharenum)
                share_vals.pop(1)
                #match names
                if row_vals[0] == share_vals[0]:
                    #cut to 20 quarters
                    trim_share = share_vals[1:21]
                    if not any(item in ('@NA', '@IF', '@NM',
                                        '@CF', '@SF', '@AF')
                           for item in trim_share):
                        #if headers
                        if row_vals[0] == 'Company Name':
                            share_count = []
                            share_price = []
                            #append new headers
                            for i in range(61):
                                share_count.append('share_count_Mtly[-'+str(i)+']')
                                share_price.append('share_price_Mtly[-'+str(i)+']')
                            row_vals.extend(share_count)
                            row_vals.extend(share_price)
                        else:
                            #stretch out the count values
                            interp_rows = [trim_share[0]]
                            for i in range(len(trim_share)-1):
                                interped = interpolate(trim_share[i], 
                                                       trim_share[i+1],2)
                                interp_rows.extend(interped)
                            interped = interpolate(trim_share[19], 
                                                   trim_share[19],2)
                            interp_rows.extend(interped)
                            row_vals.extend(interp_rows)

                            #share price = market val / no. of shares
                            share_list = []  
                            for i in range(1,62):
                                share_list.append(row_vals[i]/row_vals[i+304])
                            row_vals.extend(share_list)

                        csvWriter.writerow(row_vals)

def parsecsv(csvname):
    table = mlab.csv2rec(csvname)
    fields = [{'name':'market_valuemnthly','cols':[]},
              {'name':'priceearnings__monthly','cols':[]},
              {'name':'pricecash_flowshare_mtly','cols':[]},
              {'name':'price_to_book','cols':[]},
              {'name':'beta','cols':[]},
              {'name':'share_count_mtly','cols':[],
               'name':'share_price_mtly','cols':[]}]
    
    for idx, col_name in enumerate(table.dtype.names):
        for field in fields:
            if col_name.startswith(field['name']):
                field['cols'].append(col_name)

    #neaten down to sixty months
    for field in fields:
        field['cols'] = field['cols'][:60]
        field['cols'].reverse()
    return table, fields

def filter_table(table, fields):
    field = fields[4]
    print "filtering", field['name']
    for col in field['cols']:
        filter_val = table[col].mean() - (table[col].std())
        mask = table[col] < filter_val
        tableslice = table[mask]

    for row in tableslice:
        print row.company_name
    print "companies", len(tableslice)
            
    return tableslice
    
def graph_fields(table, fields):
    for field in fields:
        print "graphing", field['name']
        results_list = []
        for record in table:
            result = []
            for column in field['cols']:
                result.append(record[column])
            results_list.append(result)
        
        graph.plot_2d(results_list, "graphs/"+field['name'])
        #graph.plot_ave(results_list, "graphs/"+field['name'])
        #graph.boxplot_data(results_list, "graphs/"+field['name'])

#exceltocsv('oct22.xlsx','oct22cleaned.csv')
TABLE, FIELDS = parsecsv('oct22cleaned.csv')
TABLE = filter_table(TABLE, FIELDS)
#graph_fields(TABLE, FIELDS)
#print interpolate(1000,2000,9)
