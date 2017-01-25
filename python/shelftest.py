import shelve

flights = {"1":"A", "2":"B", "3":"C"}
times = ["230pm", "320pm", "420pm"]

db = shelve.open("shelved.dat", "n")

db['flights'] = flights
db['times'] = times

print db.keys()

db.close()

f = open("shelved.dat", "r")
data = f.read()
print data
f.close()

# Retrieving Objects from a Shelve File

db = shelve.open("shelved.dat", "r")

for k in db.keys():
    obj = db[k]
    print "%s: %s" % (k, obj)

flightDB = db['flights']
flights = flightDB.keys()
cities = flightDB.values()
times = db['times']

x = 0
for flight in flights:
    print ("Flight %s leaves for %s at %s" % (flight, cities[x],  times[x]))
    x+=1


db.close()
