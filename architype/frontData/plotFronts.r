baseDir <- "/home/jonathan/Jonathan/programs/femtastic/frontData"

#This will compare the average population fitness for a run of experiments

plotFronts <- function()
{
  files = list.files(path=baseDir, pattern = "Front[0-9]+")
  noOfRuns <- length(files)
  xCoords <- c()
  yCoords <- c()

  postscript(file="final.ps", paper="special",
  width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
  horizontal=FALSE)
  colCount <-1
  title = "pareto front"
  plot(400,400,col=0,xlim=c(200,10000000000),ylim=c(0,1000),main=title,xlab="stresses(kN)",ylab="no. of beams")
  for(file in files)
    {
      print(file)
      title = paste("front",colCount,sep=" ")
      localFile = paste(baseDir,file, sep = "/")
      tmpExperiment <- read.table(localFile);
      xCoords <- tmpExperiment$V1
      yCoords <- tmpExperiment$V2
      points(xCoords,yCoords,col=colCount)
      colCount <-colCount+1
    }
  dev.off()
}
plotFronts()
