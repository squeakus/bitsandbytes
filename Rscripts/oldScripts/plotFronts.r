#This will compare the average population fitness for a run of experiments

plotFronts <- function()
{
  files = list.files( pattern = "Front[0-9]+")
  noOfRuns <- length(files)
  xCoords <- c()
  yCoords <- c()

  postscript(file="final.ps", paper="special",
  width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
  horizontal=FALSE)
  colCount <-1
  title = "pareto front"
  plot(400,400,col=0,xlim=c(0,1000),ylim=c(0,500),main=title,xlab="stresses(kN)",ylab="no. of beams")
  for(file in files)
    {
      print(file)
      title = paste("front",colCount,sep=" ")
      tmpExperiment <- read.table(file);
      xCoords <- tmpExperiment$V1
      yCoords <- tmpExperiment$V2
      points(xCoords,yCoords,col=colCount)
      colCount <-colCount+1
    }
  dev.off()
}
plotFronts()
