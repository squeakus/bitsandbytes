#baseDir <- "/home/jonathan/Jonathan/programs/femtastic/frontData"
baseDir <- getwd()
#This will compare the average population fitness for a run of experiments

plotMultiFronts <- function()
{
  files = list.files(path=baseDir, pattern = "Pop[0-9]+")
  noOfRuns <- length(files)
  xCoords <- c()
  yCoords <- c()
  
  postscript(file="gens.ps", paper="special",
  width=8, height=8, onefile=TRUE, encoding="TeXtext.enc",
  horizontal=FALSE)
  par(mfrow=c(3,3))
  genCount <- 1
  for(file in files)
    {
      print(file)
      title = paste("generation:",genCount,sep=" ")
      localFile = paste(baseDir,file, sep = "/")
      tmpExperiment <- read.table(localFile);
      
      xCoords <- tmpExperiment$V1
      yCoords <- tmpExperiment$V3
      plot(xCoords,yCoords,xlim=c(0,max(xCoords)), ylim=c(0, max(yCoords)), pch=16,main=title,xlab="stresses(kN)",ylab="no. of beams")
      genCount <-genCount+1
    }
  dev.off()
}
plotMultiFronts()
