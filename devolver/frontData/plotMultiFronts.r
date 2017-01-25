#This will compare the average population fitness for a run of experiments
require("gplots")

plotMultiFronts <- function()
{
  files = list.files(pattern = "Gen[0-9]+")
  noOfRuns <- length(files)
  xCoords <- c()
  yCoords <- c()

  postscript(file="gen.ps", paper="special",
  width=8, height=8, onefile=TRUE, encoding="TeXtext.enc",
  horizontal=FALSE)
  par(mfrow=c(3,3))
  genCount <- 1
  for(file in files)
    {
      print(file)
      title = paste("generation:",genCount,sep=" ")
      tmpExperiment <- read.table(file);
      xCoords <- tmpExperiment$V1
      yCoords <- tmpExperiment$V2
      bandplot(xCoords,yCoords,xlim=c(0,500),ylim=c(0,200),pch=16,main=title,xlab="stresses(kN)",ylab="no. of beams")
      genCount <-genCount+1
    }
  dev.off()
}
plotMultiFronts()
