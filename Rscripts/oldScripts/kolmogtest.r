require("gplots")
# This will take the experiments passed in and graph the mean of the best reults


kolmogtest <- function()
{
	
	#baseDir <- "/Users/jonathanbyrne/results/maxAdjusted"
        experimentDir <- "/Users/jonathanbyrne/results/maxAdjusted/max_max_depth8_xo_0.0_struct_m_0.01"

        #no. of files to read from
        noOfRuns <- 550
        
	#create alist of all the files in the folder
        files = list.files(path=experimentDir, pattern = ".dat")		
	cnt <- 0               
        #create vector to store results
        histArray <- matrix(NaN,nrow=noOfRuns)  #this has a row for each run

        #take last element from the run to do histogram
	for(file in files)
	{
          cnt <- cnt+1   
          localFile = paste(experimentDir,file, sep = "/")
          tmpExperiment <- read.table(localFile); 
          
          histArray[cnt] <- tail(tmpExperiment$V1, n=1)   #add the first row(best) to the column
	}
        ks.test(histArray,pnorm)        
}
