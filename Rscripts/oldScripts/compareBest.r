require("gplots")
require("Hmisc")
require("matrixStats")
# This will take the experiments passed in and graph the mean of the best reults


compareBest <- function(experimentArray)
{
	
	#baseDir <- "/Users/jonathanbyrne/results/maxAdjusted"
        baseDir <- "/home/jonathan/results/maxAdjusted"
        
	noOfExperiments <- length(experimentArray)
        Noofgens <- 50 + 1
	#create an array to contain the results, a column for each experiment
	compareArray <- matrix(,nrow=noOfGens,ncol=noOfExperiments)
        stdDevArray <- matrix(,nrow=noOfGens,ncol=noOfExperiments)
	for(i in 1:noOfExperiments)
	{
		
		Experimentdir <- paste(baseDir, experimentArray[i] , sep="/")
		files = list.files(path=experimentDir, pattern = ".dat")		
		noOfRuns <- length(files)
		cnt <- 0               
		bestArray <- matrix(NaN,nrow=noOfGens,ncol=noOfRuns)  #this has a column for each run
    
		for(file in files)
		{
 		  cnt <- cnt+1   
 		  localFile = paste(experimentDir,file, sep = "/")
  		  tmpExperiment <- read.table(localFile); 
		 
  		  bestArray[,cnt] <- tmpExperiment$V1   #add the first row(best) to the column
		}
                #get the log!
		#bestArray <- log10(bestArray)

                #calculate the mean and stdDev for each generation
		meanBest <- rowMeans(bestArray)  
                stdDev <- rowSds(bestArray)
                compareArray[,i] <- meanBest
                stdDevArray[,i] <- stdDev
              }

    #matplot will plot each column separately
	matplot(compareArray,type ="l",ylim = c(min(compareArray),max(compareArray)), main="medium Target",col=c(1:5),lty=1,xlab ="generation", ylab= "fitness")

        for(j in 1:noOfExperiments)
	{
         for(i in seq(5,noOfGens,10))
           {
              errbar(i+j,compareArray[i+j,j],compareArray[i+j,j]+stdDevArray[i+j,j],compareArray[i+j,j]-stdDevArray[i+j,j],,,add="true")
           }
        }
         smartlegend(x="right", y="top",names(experimentArray),col=c(1:5),lty=1, inset=0)
	
#	matplot(compareArray,type ="l",ylim = c(0, 250), main="Best Fitness(SteadyState)", xlab ="generation", ylab= "fitness")

}
