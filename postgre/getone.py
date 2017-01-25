import psycopg2
import sys

con = None

try:
     
    con = psycopg2.connect(database='testdb', user='janbodnar') 
    
    cur = con.cursor()     
    cur.execute("SELECT * FROM cars")

    while True:
      
        row = cur.fetchone()
        
        if row == None:
            break
            
        print row[0], row[1], row[2]
    

except psycopg2.DatabaseError, e:
    print 'Error %s' % e    
    sys.exit(1)
    
    
finally:
    
    if con:
        con.close()
