baseDir <- "/home/jonathan/Jonathan/programs/femtastic"

#This will compare the average population fitness for a run of experiments

plotFronts <- function()
{
  files = list.files(path=baseDir, pattern = "Front[0-9]+")
  noOfRuns <- length(files)
  xCoords <- c()
  yCoords <- c()

  postscript(file="moo.ps", paper="special",
  width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
  horizontal=FALSE)
  for(file in files)
    {
      localFile = paste(baseDir,file, sep = "/")
      tmpExperiment <- read.table(localFile);
      xCoords <- c(xCoords,tmpExperiment$V1)
      yCoords <- c(yCoords,tmpExperiment$V2)      
    }
  front <- matrix(NaN,2,length(xCoords))
  print("length")
  print(length(xCoords))
  print(length(yCoords))
  plot(xCoords,yCoords,xlab ="generation", ylab= "fitness")
}
