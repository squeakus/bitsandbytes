baseDir <- "/Users/jbyrne/MYGEVA/results"

#This plots the best and average results for a single run to the same graph

experimentPlot <- function(experiment)
{
        generations <- 201
	experimentDir <- paste(baseDir, experiment , sep="/")
	print(experimentDir);
	files = list.files(path=experimentDir, pattern = ".dat")
        noOfRuns <- length(files)
	cnt <- 0               
	averageArray <- matrix(NaN,nrow=generations,ncol=noOfRuns)  #this has a column for each experiment 

	for(file in files)
	{
 	  cnt <- cnt+1   	 
      localFile = paste(experimentDir,file, sep = "/")
 	  tmpExperiment <- read.table(localFile);
 	  averageArray[,cnt] <- tmpExperiment$V2   #add the second row(average) to the column
	}
	meanAverage <- rowMeans(averageArray)

	cnt <- 0               
	bestArray <- matrix(NaN,nrow=201,ncol=30)  #this has a column for each experiment 

	for(file in files)
	{
 	  cnt <- cnt+1   	 
 	  localFile = paste(experimentDir,file, sep = "/")
  	  tmpExperiment <- read.table(localFile);
  	  bestArray[,cnt] <- tmpExperiment$V1   #add the second row(average) to the column
	}
	meanBest <- rowMeans(bestArray)
   
    experimentMean <- matrix(,nrow=201, ncol=2)
	experimentMean[,1] <-meanBest
	experimentMean[,2] <-meanAverage
	matplot(experimentMean,type ="l",ylim = c(0, 100), main="Best and Average Fitness", xlab ="generation", ylab= "fitness")
		
}


