require("gplots")
require("Hmisc") # for error bars
# Generates the average of all the datfiles in a folder and
# their respective standard deviations

folderMean <- function(folder='none')
{
  genCol = 1 # index of generations
  resCol = 2 # results column
  title = "result"
  # find current folder
  if(folder == 'none'){
      folder <- getwd()
    }
  print(sprintf("graphing folder:%s",folder))
  
  # define variables
  files = list.files(path=folder,pattern="*.dat")
  noOfRuns = length(files)
  fullName = paste(folder, files[1], sep='/')
  sampleIndiv <- read.table(fullName,sep=",");
  noOfGens <- length(sampleIndiv[,genCol])
  resultArray <- matrix(NaN,nrow=noOfRuns, ncol=noOfGens)

  # read the dat files
  run = 0
  for(file in files)
  {
    run <- run + 1
    fullName = paste(folder, file, sep='/')
    result <- read.table(fullName,sep=",")
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
  matplot(mean,type ="l", main=title,
          lty=1,xlab ="generation", ylab= "fitness")
   #add std dev from results
  for(i in seq(5,noOfGens,100))
    {
      errbar(i,mean[i],mean[i]+stdDev[i],mean[i]-stdDev[i],add="true")
    }

  dev.off()#close output file
  #result = list(Mean=mean,StdDev=stdDev)
  #print("RESULT")
  #return(result)
}

