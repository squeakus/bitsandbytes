require("gplots")
# generate histogram of results
histogram <- function(filePattern)
{
        experimentDir <- getwd()       
	#create alist of all the files in the folder
        files = list.files(path=experimentDir, pattern = filePattern)
	cnt <- 0
        #no. of files to read from
        noOfRuns <- length(files)
        print(sprintf("no. of files: %d",noOfRuns))
        #create vector to store results
        histArray <- matrix(NaN)  #this has a row for each run
        
        #take last element from the run to do histogram
	for(file in files)
	{
          cnt <- cnt+1   
          localFile = paste(experimentDir,file, sep = "/")
          tmpExperiment <- read.table(localFile); 
          histArray <- append(histArray, tmpExperiment$V3)   #add the last element(tail)
	}
        cleanedArray = histArray[histArray > 0]
        filename = paste(filePattern,'ps',sep='.')
        postscript(file=filename, paper="special",width=8,
                   height=8, onefile=TRUE, encoding="TeXtext.enc",
                   horizontal=FALSE)
        myseq <- seq(5,100,5)
        print(myseq)
        hist(cleanedArray, xlim = c(0,800), breaks=seq(0,800,5))        
        dev.off()
      }
histogram('nodal')
histogram('struct')
