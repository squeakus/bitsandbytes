#!/usr/bin/env R --vanilla

#This should be used in conjunction with the makeAOVarray.pl script
#This will analyse the results in conjunction with the parameters 
#you pass in to see if they make and impact statistically

dirname <- c("/Users/jonathanbyrne/results")

labelname <- ""

tablename <- paste(dirname, sep="/", "collated.dat")

## read data, and set up column names and factors
gdata <- read.table(tablename)

#names(gdata) = c("index","mutOp", "crossover","mutRate","result")
names(gdata) = c("index","mutOp","result")
mutOp<-factor(gdata$mutOp)
result<-factor(gdata$result)

print(levels(mutOp))
print(levels(result))

## two-way anova 
#This analysis the factors together as well as separately
#ganova<-aov(gdata$result ~ mutOp*crossover*mutRate)
ganova<-aov(gdata$result ~ mutOp)

print(summary(ganova))

print("T-test results")
#print(pairwise.t.test(gdata$result,mutOp:mutRate:crossover))
print(pairwise.t.test(gdata$result,mutOp,pool.sd = TRUE, paired = FALSE))




