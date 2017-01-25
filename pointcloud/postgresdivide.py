import pandas.io.sql as psql
import psycopg2
import time

starttime = time.time()

con = psycopg2.connect(database='dublin', user='jonathan')
cur = con.cursor()
cur.execute('select max(xcoord) from points1')
maxx = cur.fetchone()[0]
cur.execute('select min(xcoord) from points1')
minx = cur.fetchone()[0]

fraction = (maxx - minx) / 5
prevx = minx
print "max", maxx
print "min", minx
print "time", time.time()-starttime

for i in range(1,5):
    curx = minx + (fraction * i)
    query = 'select * from points1 where xcoord > '+str(prevx)+' and xcoord < '+str(curx)
    group = psql.frame_query(query, con=con)
    print "group",i,len(group),"time", time.time()-starttime
    group.to_csv("damex"+str(i)+".xyz", sep=' ', header=False, index=False)
    prevx = curx


con.close()
#for point in points['xcoord']:
#    print point
