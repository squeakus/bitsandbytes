require("gplots")
require("Hmisc")
require("matrixStats")
# This will take the experiments passed in and graph the mean of the best reults


epsCompareResult <- function(experimentArray)
{
	fileName <- "dTreeLengthDepth100"
        names(experimentArray) <- c("standard","nodal","structural","subtree")
        graphTitle <- "Average derivation tree depth for depth 100"
        #baseDir <- "/Users/jonathanbyrne/results/maxAdjusted"
        baseDir <- "/Users/jonathanbyrne/results/bigDepth2"
        outputDir <- "/Users/jonathanbyrne/results"
        
        fullFileName <- paste(outputDir,fileName, sep="/")
    	fullFileName <- paste(fullFileName, "eps", sep=".")
	
        
	noOfExperiments <- length(experimentArray)
        noOfGens <- 50 + 1
        noOfRuns <- 500
	#create an array to contain the results, a column for each experiment
	compareArray <- matrix(,nrow=noOfGens,ncol=noOfExperiments)
        stdDevArray <- matrix(,nrow=noOfGens,ncol=noOfExperiments)
	for(i in 1:noOfExperiments)
	{
		
		experimentDir <- paste(baseDir, experimentArray[i] , sep="/")
		files = list.files(path=experimentDir, pattern = ".dat")		
		cnt <- 0               
		bestArray <- matrix(NaN,nrow=noOfGens,ncol=noOfRuns)  #this has a column for each run
    
		for(file in files)
		{
 		  cnt <- cnt+1   
 		  localFile = paste(experimentDir,file, sep = "/")
  		  tmpExperiment <- read.table(localFile); 
		 
  		  bestArray[,cnt] <- tmpExperiment$V8   #add the first row(best) to the column
		}
                #get the log!
		#bestArray <- log10(bestArray)

                #calculate the mean and stdDev for each generation
		meanBest <- rowMeans(bestArray)  
                stdDev <- rowSds(bestArray)
                compareArray[,i] <- meanBest
                stdDevArray[,i] <- stdDev
	}
  
        
    print(fullFileName)

    postscript(file=fullFileName, paper="special",
    width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
    horizontal=FALSE)
        
    #matplot will plot each column separately
    matplot(compareArray,type ="l",ylim = c(1,50), main=graphTitle,col=1,lty=c(1,2,5,6),xlab ="generation", ylab= "tree depth")

#        for(j in 1:noOfExperiments)
#	{
#         for(i in seq(5,noOfGens,20))
#           {             
#             if((compareArray[i+j,j]-stdDevArray[i+j,j]) >0)
#               {
#              errbar(i+j,compareArray[i+j,j],compareArray[i+j,j]+stdDevArray[i+j,j],compareArray[i+j,j]-stdDevArray[i+j,j],,,add="true",lty=1)
#               }
#            }
#        }
         smartlegend(x="right", y="top",names(experimentArray),col=1,lty=c(1,2,5,6), inset=0)

        dev.off()
    	fullFileName <-""
}
