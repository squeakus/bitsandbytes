require("gplots")
require("Hmisc")
require("matrixStats")
# This will take the experiments passed in and graph the mean of the best reults


singleBest <- function()
{

	  files = list.files( pattern = "Pop[0-9]+.dat")		
	  noOfGens <- length(files)
	  gen <- 0               
	  bestLength <- matrix(NaN,nrow=noOfGens,ncol=1)  #this has a column for each run
          bestStress <- matrix(NaN,nrow=noOfGens,ncol=1)
          avrLength <- matrix(NaN,nrow=noOfGens,ncol=1)
          avrStress <- matrix(NaN,nrow=noOfGens,ncol=1)
	  for(file in files)
	  {
            print(file)
            gen <- gen+1   
            tmpExperiment <- read.table(file);
            bestLength[gen,] <- min(tmpExperiment$V1)
            bestStress[gen,] <- min(tmpExperiment$V2)
            avrLength[gen,] <- mean(tmpExperiment$V1)
            avrStress[gen,] <- mean(tmpExperiment$V2)
            
	  }
          print("Length")
          print(bestLength)
          print("Stress")
          print(bestStress)
          print("Avr Length")
          print(avrLength)
          print('AvrStress')
          print(avrStress)
          allArrays <- cbind(bestLength,bestStress,avrLength,avrStress)
          matplot(allArrays, type='l')
          #calculate the mean and stdDev for each generation
	  #meanBest <- rowMeans(bestArray)  
          #stdDev <- rowSds(bestArray)
          #compareArray[,i] <- meanBest
          #stdDevArray[,i] <- stdDev
          

    #matplot will plot each column separately
##      matplot(compareArray,type ="l",ylim = c(min(compareArray),max(compareArray)), main="medium Target",col=c(1:5),lty=1,xlab ="generation", ylab= "fitness")

##         for(j in 1:noOfExperiments)
## 	{
##          for(i in seq(5,noOfGens,10))
##            {
##               errbar(i+j,compareArray[i+j,j],compareArray[i+j,j]+stdDevArray[i+j,j],compareArray[i+j,j]-stdDevArray[i+j,j],,,add="true")
##            }
##         }
##          smartlegend(x="right", y="top",names(experimentArray),col=c(1:5),lty=1, inset=0)
	
#	matplot(compareArray,type ="l",ylim = c(0, 250), main="Best Fitness(SteadyState)", xlab ="generation", ylab= "fitness")

}
