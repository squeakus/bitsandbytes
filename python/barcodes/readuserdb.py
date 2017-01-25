userfile = open("User.txt", 'r') 

for line in userfile:
    row = line.split(';')
    genus = row[0].replace('"','')
    species = row[1].replace('"','')
    variety = row[2].replace('"','')
    barcode = row[8].replace('"','')
    #print "genus",genus,"species",species,"var",variety, "bar",
    if len(barcode) < 12:
        print "genus",genus,"species",species,"var",variety, "HAS SEMICOLON"

     # if len(row[]) < 1:
     #    print "genus",genus,"species",species,"var",variety, "HAS NO PICTURE" 

    if len(row[4]) < 1:
        print "genus",genus,"species",species,"var",variety, "HAS NO DESCRIPT"
