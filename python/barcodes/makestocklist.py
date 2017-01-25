stockfile = open("stock1.txt", 'w') 

while True:
    barcode = raw_input()
    if not barcode == '':
        stockfile.write("{'code':'"+barcode+"','name':'PUTNAMEHERE','price':PUTPRICEHERE}\n")
        
