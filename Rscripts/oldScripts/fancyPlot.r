d <- read.table(textConnection("id x y
1  10  500
1  15  300
1  23  215
1  34  200
2  5    400
2  13  340
2  15  210
3  10  200
3  12  150
3  16  30"), head=TRUE)

str(d)

print("loaded table")
print(d)

library(ggplot2)

p <-
ggplot(d) +
 geom_path(aes(x,y, group=id))

qplot(p)
p ## grouping withouth legend

## colour based on id
p + aes(colour=factor(id))

## linetype based on id
p + aes(linetype=factor(id))
