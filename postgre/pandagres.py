import pandas.io.sql as psql
import psycopg2

con = psycopg2.connect(database='testdb', user='jonathan')
points = psql.frame_query('select * from dublin', con=con)
print "loaded points from db", len(points)
con.close()

for point in points['xcoord']:
    print point
