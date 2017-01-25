stockfile = open("stock.txt", 'r') 
stocklist = []

for line in stockfile:
    item = eval(line)
    stocklist.append(item)

while True:
    barcode = raw_input()
    if not barcode == '':
        for item in stocklist:
            if item['code'] == barcode:
                print "item:", item['name'], " price:", item['price']
        
