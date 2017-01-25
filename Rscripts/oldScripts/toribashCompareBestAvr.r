require("gplots")
require("Hmisc")
require("matrixStats")
require("abind")
# This will take the experiments passed in and graph the mean of the best reults


toriCompareBoth <- function(experimentArray)
{
	
	#baseDir <- "/Users/jonathanbyrne/results/maxAdjusted"
        baseDir <- "/Users/jonathanbyrne/results"
        
	noOfExperiments <- length(experimentArray)
        noOfGens <- 51 
        
	#create an array to contain the results, a column for each experiment
	compareArray <- matrix(,nrow=noOfGens,ncol=noOfExperiments)
        meanCompareArray <- matrix(,nrow=noOfGens,ncol=noOfExperiments)
        stdDevArray <- matrix(,nrow=noOfGens,ncol=noOfExperiments)
	for(i in 1:noOfExperiments)
	{
		
		experimentDir <- paste(baseDir, experimentArray[i] , sep="/")
		files = list.files(path=experimentDir, pattern = ".dat")
		cnt <- 0
                noOfRuns <- length(files)
		bestArray <- matrix(NaN,nrow=noOfGens,ncol=noOfRuns)  #this has a column for each run
                averageArray <- matrix(NaN,nrow=noOfGens,ncol=noOfRuns)  #this has a column for each run
    
		for(file in files)
		{
                  print(file)
 		  cnt <- cnt+1   
 		  localFile = paste(experimentDir,file, sep = "/")
  		  tmpExperiment <- read.table(localFile); 
		  #print(tmpExperiment$V2)
                 # print(bestArray[,cnt])
  		  bestArray[,cnt] <- tmpExperiment$V2   #add the first row(best) to the column
                  averageArray[,cnt] <- tmpExperiment$V3   #add the second row(average) to the column
		}
                #get the log!
		#bestArray <- log10(bestArray)

                #calculate the mean and stdDev for each generation
		meanBest <- rowMeans(bestArray)
                meanAverage <- rowMeans(averageArray)
                stdDev <- rowSds(bestArray)
                compareArray[,i] <- meanBest
                meanCompareArray[,i] <-meanAverage
                stdDevArray[,i] <- stdDev
                totalArray = abind(compareArray,meanCompareArray)
	}
        print(tail(stdDevArray,n=1))
    #matplot will plot each column separately
	matplot(totalArray,type ="l",ylim = c(min(totalArray),max(totalArray)), main="bestFitness",col=c(1:5),lty=1,xlab ="generation", ylab= "fitness")

        for(j in 1:noOfExperiments)
	{
         for(i in seq(5,noOfGens,10))
           {
              errbar(i+j,compareArray[i+j,j],compareArray[i+j,j]+stdDevArray[i+j,j],compareArray[i+j,j]-stdDevArray[i+j,j],,,add="true")
           }
        }
         smartlegend(x="left", y="top",names(experimentArray),col=c(1:5),lty=1, inset=0)
	
#	matplot(compareArray,type ="l",ylim = c(0, 250), main="Best Fitness(SteadyState)", xlab ="generation", ylab= "fitness")

}
