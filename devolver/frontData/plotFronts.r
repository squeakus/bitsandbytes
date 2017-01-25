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
  colCount <- 0
  #niceColors <- rainbow(28,start=0)
  niceColors <- heat.colors(40, alpha = 1)
  title = "pareto fronts coded by color"
  plot(400,400,col=0,xlim=c(0,750),ylim=c(0,300),main=title,pch=16,xlab="Normalised Stress",ylab="No. of Beams")
  print(niceColors)
  for(file in files)
    {
      print(file)
      title = paste("front",colCount,sep=" ")
      tmpExperiment <- read.table(file);
      xCoords <- tmpExperiment$V1
      yCoords <- tmpExperiment$V2
      print(colCount)
      points(xCoords,yCoords,col=niceColors[colCount])
      colCount <-colCount+1
    }
  dev.off()
}
plotFronts()
