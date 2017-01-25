require("gplots")
#This will create a boxplot for the best results from a experimental run

baseDir <- "/Users/jonathanbyrne/results"

boxBest <- function(experimentArray)
{
  #automatically setting up boxPlot Array
  noOfExperiments <- length(experimentArray)
  experimentDir <- paste(baseDir, experimentArray[1] , sep="/")
  files = list.files(path=experimentDir, pattern = ".dat")
  noOfRuns = length(files)
  
  #creating array
  boxBestArray <- matrix(NaN,ncol=noOfExperiments,nrow=noOfRuns)
  for(i in 1:noOfExperiments)
    {		
      experimentDir <- paste(baseDir, experimentArray[i] , sep="/")
      files = list.files(path=experimentDir, pattern = ".dat")		
      cnt <- 0
      
      for(file in files)
        {
          cnt <- cnt+1   	 
          localFile = paste(experimentDir,file, sep = "/")
          tmpExperiment <- read.table(localFile);
          
          boxBestArray[cnt,i] <- tail(tmpExperiment$V2, n=1)   #add the first row(best) to the column
        } 
    }	
  bp <- boxplot(boxBestArray)
  print(bp)
  print(ave(boxBestArray[,1]))
  print(ave(boxBestArray[,2]))
}
