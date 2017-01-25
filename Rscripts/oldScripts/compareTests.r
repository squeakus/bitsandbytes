require("gplots")

# This will take the experiments passed in and graph the mean of the best reults

#baseDir <- "/Users/jbyrne/MYGEVA/PRCConcatResults"
baseDir1 <- "/Users/jbyrne/MYGEVA/ExperimentManager/ge_op_jb"
baseDir2 <- "/Users/jbyrne/MYGEVA/ExperimentManager/intflipres"
compareTests <- function(experimentArray1,experimentArray2)
{
	
	noOfExperiments <- length(experimentArray) + length(experimentArray2)
	print(noOfExperiments)
	#create an array to contain the results, a column for each experiment
	compareArray <- matrix(,nrow=51,ncol=noOfExperiments)
	for(i in 1:noOfExperiments)
	{
	if(i <= length(experimentArray))
	{	
		experimentDir <- paste(baseDir1, experimentArray1[1] , sep="/")
		files = list.files(path=experimentDir, pattern = ".dat")
	}
	else
	{
	    experimentDir <- paste(baseDir2, experimentArray2[1] , sep="/")
		files = list.files(path=experimentDir, pattern = ".dat")
	}
		print(experimentDir)
		cnt <- 0               
		bestArray <- matrix(NaN,nrow=51,ncol=30)  #this has a column for each run
    
		for(file in files)
		{
 		  cnt <- cnt+1   	 
 		  localFile = paste(experimentDir,file, sep = "/")
  		  tmpExperiment <- read.table(localFile);
  		  bestArray[,cnt] <- tmpExperiment$V1   #add the first row(best) to the column
		}
		
		#calculate the mean for each generation
		meanBest <- rowMeans(bestArray)      
        compareArray[,i] <- meanBest 
	}	

    #matplot will plot each column separately
	matplot(compareArray,type ="l",ylim = c(0,30), main="Best Fitness without Crossover (Symbolic Regression)",col="black",lty= 1:4,xlab ="generation", ylab= "fitness")
	
#	matplot(compareArray,type ="l",ylim = c(0, 250), main="Best Fitness(SteadyState)", xlab ="generation", ylab= "fitness")

}
