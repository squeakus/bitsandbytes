import xlrd, csv, graph
import matplotlib.mlab as mlab
import numpy as np

def exceltocsv(spreadsheet, csvname):
    wb = xlrd.open_workbook(spreadsheet)
    wb2 = xlrd.open_workbook('shareQuantity.xlsx')
    
    sh = wb.sheet_by_index(0)
    sh2 = wb2.sheet_by_index(0)
    
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
            for sharenum in range(1,sh2.nrows):
                share_vals = sh2.row_values(sharenum)
                share_vals.pop(1)

                if row_vals[0] == share_vals[0]:
                    trim_share = share_vals[1:21]
                    if not any(item in ('@NA', '@IF', '@NM',
                                        '@CF', '@SF', '@AF')
                           for item in trim_share):
                        new_row = row_vals.extend(trim_share)
                        csvWriter.writerow(row_vals)

def parsecsv(csvname):
    table = mlab.csv2rec(csvname)
    print "length", len(table)
    fields = [{'name':'market_valuemnthly','cols':[]},
              {'name':'priceearnings__monthly','cols':[]},
              {'name':'pricecash_flowshare_mtly','cols':[]},
              {'name':'price_to_book','cols':[]},
              {'name':'beta','cols':[]}]
    #{'name':'com_shares_outstanding_qtly','cols':[]}]
    
    for idx, col_name in enumerate(table.dtype.names):
        for field in fields:
            if col_name.startswith(field['name']):
                field['cols'].append(col_name)

    for field in fields:
        share_stretched = []
        if field['name'] == 'com_shares_outstanding_qtly':
            for col in field['cols']:
                share_stretched.append(col)
                share_stretched.append(col)
                share_stretched.append(col)
                share_stretched.append(col)
            field['cols'] = share_stretched
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
    print len(tableslice)
            
    return tableslice
    
def graph_attrib(table, fields):
    for field in fields:
        print "graphing", field['name']
        results_list = []
        for record in table:
            result = []
            for column in field['cols']:
                result.append(record[column])
            results_list.append(result)
        
        graph.plot_2d(results_list, field['name'])
        graph.plot_ave(results_list, field['name'])
        graph.boxplot_data(results_list, field['name'])

#exceltocsv('oct22.xlsx','oct22cleaned.csv')
TABLE, FIELDS = parsecsv('oct22cleaned.csv')
TABLE = filter_table(TABLE, FIELDS)
graph_attrib(TABLE, FIELDS)
