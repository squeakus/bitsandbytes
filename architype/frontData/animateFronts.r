#baseDir <- "/home/jonathan/Jonathan/programs/femtastic/frontData"
baseDir <- getwd()
#This will compare the average population fitness for a run of experiments

plotMultiFronts <- function()
{
  files = list.files(path=baseDir, pattern = "Pop[0-9]+")
  noOfRuns <- length(files)
  xCoords <- c()
  yCoords <- c()
  
  genCount <- 1
  for(file in files)
    {
      print(file)
      title = paste("generation:",genCount,sep=" ")
      localFile = paste(baseDir,file, sep = "/")
      tmpExperiment <- read.table(localFile);
      
      xCoords <- tmpExperiment$V1
      yCoords <- tmpExperiment$V3
      if(genCount == 1){
        maxXCoords <- max(xCoords)
        maxYCoords <- max(yCoords)
      }
      filename = paste("gen",sprintf("%03d",genCount),sep ="")
      filename = paste(filename,"eps",sep =".")

      postscript(file=filename, paper="special",
                 width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
                 horizontal=FALSE)

      
      print(filename)

      plot(xCoords,yCoords,xlim=c(0,maxXCoords), ylim=c(0, maxYCoords),
           pch=16,main=title,xlab="Stresses (kN)",ylab="Total Weight (kg)")
      genCount <-genCount+1
      dev.off()
    }
}
plotMultiFronts()
