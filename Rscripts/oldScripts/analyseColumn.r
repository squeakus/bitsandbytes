require("gplots")
require("Hmisc") # for error bars
# This will break down the results for each experiment passed into it.
# It will display the best and average from the end of the run with 
# their respective standard deviations

analyseResults <- function()
{
  genCol = 1 # index of generations
  resCol = 2 # results column
  yMax = 5000000
  yMin = 100000
  # find the variables
  baseDir <- getwd()
  files = list.files(pattern="*.dat")
  noOfRuns = length(files)
  result <- read.table(files[1],sep=",");
  noOfGens <- length(result[,genCol])
  print(noOfGens)
  resultArray <- matrix(NaN,nrow=noOfRuns, ncol=noOfGens)

  # read the dat files
  run = 0
  for(file in files)
  {
    run <- run + 1   	 
    result <- read.table(file,sep=",")
    resultArray[run,] <- result[,resCol]  
  }
  #calculate mean and standard Dev
  mean <- colMeans(resultArray)
  stdDev <- sd(resultArray)

  # output to file
  postscript(file='result.ps', paper="special",width=8,
             height=8, onefile=FALSE, encoding="TeXtext.enc",
             horizontal=FALSE)
  
  #plot mean of the runs
  matplot(mean,type ="l", main="tournament30", ylim = c(yMin,yMax),
          lty=1,xlab ="generation", ylab= "fitness")

  #add std dev from results
  for(i in seq(5,noOfGens,100))
    {
      errbar(i,mean[i],mean[i]+stdDev[i],mean[i]-stdDev[i],add="true")
    }
  dev.off()#close output file
}	
	
analyseResults()
