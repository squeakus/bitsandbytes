require("gplots")
# Generates the average of all the datfiles in a folder and
# their respective standard deviations

processFolder <- function(folder='none')
{
  genCol = 1 # index of generations
  resCol = 2 # results column
  title = "result"

  # find current folder
  if(folder == 'none'){
      folder <- getwd()
    }
  print(sprintf("processing folder:%s",folder))
  
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
  result = list(Mean=mean,StdDev=stdDev)
  return(result)
}	
