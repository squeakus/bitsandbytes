import csv, re
import pyExcelerator as excel

float_regex = re.compile('\d+(\.\d+)?')
alpha_regex = re.compile('[a-zA-Z]')
atsign = re.compile('@')


def integerise(row):
    newRow = []
    for cell in row:
        cell = cell.replace(',','')
        cell = cell.replace('\'','')
        cell = cell.replace('(', '').replace(')', '')
        if atsign.search(cell):
            cell = ''

        if not alpha_regex.search(cell):
            if float_regex.match(cell):
                cell = float(cell)

        newRow.append(cell)
    return newRow

def parse_csv(csv_file):
    results = []
    reader = csv.reader(open(csv_file,'rU'),delimiter=',')
    count = 0

    for entry in reader:
        entry = integerise(entry)
        results.append(entry)

    return results

def output_csv(results):
    csvWriter = csv.writer(open('cleaned.csv', 'wb'), delimiter=',', quoting=csv.QUOTE_NONE)
    for row in results:
        csvWriter.writerow(row)

def output_excel(results):
    work_book = excel.Workbook()
    doc_sheet = work_book.add_sheet("sheet1")

    for rowID, row in enumerate(results):
        for cellID, cell in enumerate(row):
            doc_sheet.write(rowID,cellID, cell)

    doc_sheet.col(0).width = 0x0d00 + 5000
    work_book.save('cleaned.xls')

RESULTS = parse_csv("oct22.csv")
output_csv(RESULTS)
#output_excel(RESULTS)
