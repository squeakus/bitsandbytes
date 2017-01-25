require("gplots")

# This will take the directory passed in and graph the mean of the best results

problems <-c("treeD","treeE","treeF","treeG")
ranges <- c(0.2,0.2,0.2,0.2)
xoRates <-c("0.0","0.9")
experimentDir <- ""
mutationOps <-c("intflip","node","struct")
mutRates <-c("0.1","0.2")
noOfGenerations <- 50 +1
noOfRuns <- 30

graphAll <- function(experiment)
{
  baseDir <- "/Users/jbyrne/ExperimentManager"	
  experimentDir <- paste(baseDir, experiment , sep="/")
  experiment <- paste(experimentDir,experiment, sep="/")
  rangeCnt <-1
  
  for(problem in problems)
  {  	
  	graphRange <- ranges[rangeCnt]
  	rangeCnt <- rangeCnt +1
  	for(xoRate in xoRates)
  	{
  	  for(mutRate in mutRates)
  	  {	
  	  	compareArray <- matrix(,nrow=noOfGenerations,ncol=length(mutationOps))
        expCounter <-1
        for(mutationOp in mutationOps)
    	  {
    		experimentName <-paste(experiment,problem,"xo_std_xo",xoRate,mutationOp,"m",mutRate, sep="_")	
    		files = list.files(path=experimentName, pattern = ".dat")
		   
			cnt <- 0               
			bestArray <- matrix(NaN,nrow=noOfGenerations,ncol=noOfRuns)  #this has a column for each run
			#print(experimentName)		
			for(file in files)
			{
			  #print(file)
	 		  cnt <- cnt+1   
	 		  localFile = paste(experimentName,file, sep = "/")
	  		  tmpExperiment <- read.table(localFile); 
	  		  bestArray[,cnt] <- tmpExperiment$V1   #add the first row(best) to the column
			}
			#calculate the mean for each generation
			meanBest <- rowMeans(bestArray)      
        	compareArray[,expCounter] <- meanBest			
  	        experimentName <- ""
  	        expCounter <-expCounter+1
  	      }
  	       fileName <- paste(experimentDir,problem, sep="/")
  	       fileName <- paste(fileName,"XO",xoRate,"Mut",mutRate,sep="_")
    	   fileName <- paste(fileName, "eps", sep=".")
    	   graphTitle <- paste(problem,"XO",xoRate,"Mut",mutRate,sep=" ")
    	   print(fileName)
    	   postscript(file=fileName, paper="special", width=8, height=8, onefile=FALSE, encoding="TeXtext.enc", 		horizontal=FALSE)
  	      matplot(compareArray,type ="l",,ylim = c(0, graphRange), main=graphTitle,col=1:5,lty= 1,xlab ="generation", ylab= "fitness")	
  	      smartlegend(x="left", y="top",mutationOps,col=1:5,lty=1, inset=0)
    dev.off()
    	fileName <-""
  	  }    	
  	}
  }  	
    	
 

}
