require("gplots")
# generate histogram of results
histogram <- function(filename)
{
        resultCol = 2
	experimentDir <- getwd()       
	#create alist of all the files in the folder
        files = list.files(path=experimentDir, pattern='*.dat')
	cnt <- 0
        #no. of files to read from
        noOfRuns <- length(files)
        print(sprintf("no. of files: %d",noOfRuns))
        #create vector to store results
        histArray <- matrix()  #this has a row for each run
        
        #take last element from the run to do histogram
	for(file in files)
	{
          cnt <- cnt+1   
          localFile = paste(experimentDir,file, sep = "/")
          tmpExperiment <- read.table(localFile);
         
          lastElem = tail(tmpExperiment[,resultCol],n=1)
          print(lastElem)
          histArray <- append(histArray, lastElem)   #add the last element(tail)
	}
        print(histArray)
        histArray = histArray[histArray > 0]
        filename = paste(filename,'ps',sep='.')
        postscript(file=filename, paper="special",width=8,
                   height=8, onefile=TRUE, encoding="TeXtext.enc",
                   horizontal=FALSE)
        #myseq <- seq(5,100,5)
        #print(myseq)
        #hist(cleanedArray, xlim = c(0,800), breaks=seq(0,800,5))
        hist(histArray)
        dev.off()
      }
histogram('histogram')
#histogram('struct')
