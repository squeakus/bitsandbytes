require("gplots")
# Generates the average of all the datfiles in a folder and
# their respective standard deviations

performWilcox <- function(datfile='wilcox.dat')
{
  folder <- getwd()
  fullName = paste(folder, datfile, sep='/')
  wilcoxTable <- read.table(fullName,sep=" ");
  values <- wilcoxTable[,1]
  print(values)
  groupings <- wilcoxTable[,2]
  print(groupings)
  sink("output.txt")
  print("Performing wilcoxon rank sum")
  print(pairwise.wilcox.test(values, groupings, paired=FALSE))

}

performWilcox()
