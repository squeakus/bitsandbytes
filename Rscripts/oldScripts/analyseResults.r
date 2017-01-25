require("gplots")

# This will break down the results for each experiment passed into it.
# It will display the best and average from the end of the run with 
# their respective standard deviations

#baseDir <- "/Users/jonathanbyrne/MYGEVA/PRCConcatResults"
baseDir <- "/Users/jonathanbyrne/ExperimentManager/noxor"

analyseResults <- function(experimentArray)
{
	noOfExperiments <- length(experimentArray)
	#this holds the best results from last generation and has a column for each run
	bestArray <- matrix(NaN,nrow=noOfExperiments,ncol=50) 
	#this holds the average results from last generation and has a column for each run
        averageArray <- matrix(NaN,nrow=noOfExperiments,ncol=100)
    
	for(i in 1:noOfExperiments)
	{		
		#making a list of all the files
		experimentDir <- paste(baseDir, experimentArray[i] , sep="/")
		files = list.files(path=experimentDir, pattern = ".dat")		
		cnt <- 0               
		for(file in files)
		{
		 
 		  cnt <- cnt+1   	 
 		  localFile = paste(experimentDir,file, sep = "/")
  		  tmpExperiment <- read.table(localFile);
  		  bestArray[i,cnt] <- tmpExperiment[51,1]  #add the first row(best) to the column
		  averageArray[i,cnt] <- tmpExperiment[51,2] #add the second row(average) to the column
		}     
	}	
	
	meanBest <- rowMeans(bestArray) 
	meanAverage <-rowMeans(averageArray)
	
	#outputting results
	for(i in 1:length(experimentArray))
	{
		avrStdDev <- sd(averageArray[i,])
		bstStdDev <- sd(bestArray[i,])
		
		print(experimentArray[i])
		print(sprintf("%.2f & %.2f & %.2f & %.2f",meanBest[i],bstStdDev,meanAverage[i],avrStdDev))
	}
}
