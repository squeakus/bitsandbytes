#This will compare the average population fitness for a run of experiments
require("gplots")

plotMultiEps <- function()
{
  files = list.files(pattern = "Gen[0-9]+")
  noOfRuns <- length(files)
  xCoords <- c()
  yCoords <- c()
  genCount <- 1
  
  for(file in files)
    {
      print(file)
      filename = paste("scatter",genCount,"ps",sep=".")
      postscript(file=filename, paper="special",width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",horizontal=FALSE)
      title = paste("generation:",genCount,sep=" ")
      tmpExperiment <- read.table(file);
      xCoords <- tmpExperiment$V1
      yCoords <- tmpExperiment$V2
      smoothScatter(xCoords,yCoords,xlim=c(0,750),ylim=c(0,300),pch=16,main=title,xlab="Normalised Stress",ylab="No. of Beams")
      genCount <-genCount+1
      dev.off()
    }
  
}
plotMultiEps()
