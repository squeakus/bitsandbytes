baseDir <- "/Users/jonathanbyrne/results"

#This will compare the average population fitness for a run of experiments

compareAverage <- function(experimentArray)
{
        generations = 50 + 1
	noOfExperiments <- length(experimentArray)
	#create an array to contain the results, a column for each experiment
	compareArray <- matrix(,nrow=generations,ncol=noOfExperiments)
        stdDevArray <- matrix(,nrow=generations,ncol=noOfExperiments)
	for(i in 1:noOfExperiments)
	{
		experimentDir <- paste(baseDir, experimentArray[i] , sep="/")
		files = list.files(path=experimentDir, pattern = ".dat")
		cnt <- 0               
		noOfRuns <- length(files)
		averageArray <- matrix(NaN,nrow=generations,ncol=noOfRuns)  #this has a column for each run
    
		for(file in files)
		{
 		  cnt <- cnt+1   	 
 		  localFile = paste(experimentDir,file, sep = "/")
  		  tmpExperiment <- read.table(localFile);
  		  averageArray[,cnt] <- tmpExperiment$V3   #add the second row(average) to the column
		}
	#calculate the mean for each generation
	meanAverage <- rowMeans(averageArray)
        stdDev <- rowSds(averageArray)
        compareArray[,i] <- meanAverage
        stdDevArray[,i] <- stdDev
	}	
	 #matplot will plot each column separately
	matplot(compareArray,type ="l",ylim = c(min(compareArray),max(compareArray))*1.1, main="average Fitness(generational)",col=c(1:5),lty=1, xlab ="generation", ylab= "fitness")
        for(j in 1:noOfExperiments)
	{
         for(i in seq(5,generations,10))
           {
              errbar(i+j,compareArray[i+j,j],compareArray[i+j,j]+stdDevArray[i+j,j],compareArray[i+j,j]-stdDevArray[i+j,j],,,add="true")
           }
         smartlegend(x="left", y="top",names(experimentArray),col=c(1:5),lty=1, inset=0)
        }
}
