require("gplots")
# This will take the experiments passed in and graph the mean of the best reults


histogram <- function()
{
	
	#baseDir <- "/Users/jonathanbyrne/results/maxAdjusted"
        experimentDir <- "/Users/jonathanbyrne/results/toribash" 
        
	#create alist of all the files in the folder
        files = list.files(path=experimentDir, pattern = ".dat")		
	cnt <- 0
        #no. of files to read from
        noOfRuns <- length(files)
        print("no. of files")
        print(noOfRuns)
        #create vector to store results
        histArray <- matrix(NaN,nrow=noOfRuns)  #this has a row for each run
        
        #take last element from the run to do histogram
	for(file in files)
	{
          cnt <- cnt+1   
          localFile = paste(experimentDir,file, sep = "/")
          tmpExperiment <- read.table(localFile); 
          
          histArray[cnt] <- tail(tmpExperiment$V2, n=1)   #add the last element(tail)
	}
        hist(histArray)
        
}
