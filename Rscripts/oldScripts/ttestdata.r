require("gplots")
# This will take the experiments passed in and graph the mean of the best reults


ttestdata <- function(experimentArray)
{
        baseDir <- "/Users/jonathanbyrne/results"

        #no. of files to read from
	experimentDir <- paste(baseDir, experimentArray[1] , sep="/")
        files = list.files(path=experimentDir, pattern = ".dat")
	noOfRuns <-length(files)
        noOfExperiments <- length(experimentArray)
                
        #create vector to store last result in each run
        tmpArray <- matrix(NaN,nrow=noOfRuns)  #this has a row for each run

        #create an array to hold the two samples you want to compare
        compareArray <- matrix(,nrow=noOfRuns,ncol=noOfExperiments)
        for(i in 1:noOfExperiments)
	{
          #specify the directory
          experimentDir <- paste(baseDir, experimentArray[i] , sep="/")
          print(experimentDir)

          #create alist of all the files in the folder
          files = list.files(path=experimentDir, pattern = ".dat")		
	  cnt <- 0
          
          #take last element from the run to do histogram
	  for(file in files)
	  {
            cnt <- cnt+1   
            localFile = paste(experimentDir,file, sep = "/")
            tmpExperiment <- read.table(localFile); 
            tmpArray[cnt] <- tail(tmpExperiment$V2, n=1)   #add the first row(best) to the column
	  }

          compareArray[,i] <-tmpArray
        }
        t.test(compareArray[,1],compareArray[,2],paired = FALSE,var.equal=FALSE)
}
