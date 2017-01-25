require("gplots")
require("Hmisc")
require("matrixStats")
# This will take the experiments passed in and graph the mean of the best reults


toriCompareBest <- function(experimentArray)
{
	
	#baseDir <- "/Users/jonathanbyrne/results/maxAdjusted"
        baseDir <- "/Users/jonathanbyrne/results"
        epsName <- "ComboBest"
	noOfExperiments <- length(experimentArray)
        noOfGens <- 51 
        
	#create an array to contain the results, a column for each experiment
	compareArray <- matrix(,nrow=noOfGens,ncol=noOfExperiments)
        stdDevArray <- matrix(,nrow=noOfGens,ncol=noOfExperiments)
	for(i in 1:noOfExperiments)
	{
		
		experimentDir <- paste(baseDir, experimentArray[i] , sep="/")
		files = list.files(path=experimentDir, pattern = ".dat")
		cnt <- 0
                noOfRuns <- length(files)
		bestArray <- matrix(NaN,nrow=noOfGens,ncol=noOfRuns)  #this has a column for each run
    
		for(file in files)
		{
                  print(file)
 		  cnt <- cnt+1   
 		  localFile = paste(experimentDir,file, sep = "/")
  		  tmpExperiment <- read.table(localFile); 
  		  bestArray[,cnt] <- tmpExperiment$V2   #add the first row(best) to the column
		}

                #calculate the mean and stdDev for each generation
		meanBest <- rowMeans(bestArray)  
                stdDev <- rowSds(bestArray)
                compareArray[,i] <- meanBest
                stdDevArray[,i] <- stdDev
	}

    print(max(compareArray)*1.4)
    epsName =paste(epsName,"eps",sep =".")    
    print(epsName)
    
    postscript(file=epsName, paper="special",
    width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
    horizontal=FALSE)


    #matplot will plot each column separately
	matplot(compareArray,type ="l",ylim = c(min(compareArray),max(compareArray)*1.3), main="Best Fitness for Move Combination",col=1,lty=c(1,2,4,5,6),xlab ="generation", ylab= "fitness")

        for(j in 1:noOfExperiments)
	{
         for(i in seq(5,noOfGens,20))
           {
              errbar(i+j,compareArray[i+j,j],compareArray[i+j,j]+stdDevArray[i+j,j],compareArray[i+j,j]-stdDevArray[i+j,j],,,add="true")
           }
        }
         smartlegend(x="left", y="top",names(experimentArray),col=1,lty=c(1,2,4,5,6), inset=0)
	
#	matplot(compareArray,type ="l",ylim = c(0, 250), main="Best Fitness(SteadyState)", xlab ="generation", ylab= "fitness")
     dev.off()
     epsName <- ""
}
