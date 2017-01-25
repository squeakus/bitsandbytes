require("gplots")

#This script counts the number of invalids for each experiment

#baseDir <- "/Users/jbyrne/MYGEVA/PRCConcatResults"
baseDir <- "/Users/jbyrne/MYGEVA/ExperimentManager/mutationTest7"

countInvalids <- function(experimentArray)
{
	noOfExperiments <- length(experimentArray)
	totalInvalids <-0
	
	for(i in 1:noOfExperiments)
	{
		experimentDir <- paste(baseDir, experimentArray[i] , sep="/")
		files = list.files(path=experimentDir, pattern = ".dat")
		
		cnt <- 0               
		bestArray <- matrix(NaN,nrow=201,ncol=30)  #this has a column for each experiment 
    
		for(file in files)
		{
		  experimentInvalids <-0
 		  cnt <- cnt+1   	 
 		  localFile = paste(experimentDir,file, sep = "/")
  		  tmpExperiment <- read.table(localFile);
  		  for(j in 1:201)
  		  {
  		  	totalInvalids <-totalInvalids + tmpExperiment[j,5] #adds up the invalids in row 5
  		  	experimentInvalids <-experimentInvalids + tmpExperiment[j,5] #adds up the invalids in row 5	
  		  	  		  
  		  } 		  
		 #print(sprintf("experiment %s has %d invalids",file,experimentInvalids))
		}
     }	
     print(totalInvalids)
print(sprintf("total invalids: %d",totalInvalids))
}