capital = 10.0
annual = 0.02 # a whopping 5%
monthly = ((1+annual)**0.08333) - 1
cumulative = capital

for i in range(12):
    cumulative = cumulative + (cumulative * monthly)

print "annual", str(capital + (capital*annual))
print "cumulative", cumulative
