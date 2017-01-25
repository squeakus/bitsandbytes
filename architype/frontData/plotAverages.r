#baseDir <- "/home/jonathan/Jonathan/programs/femtastic/frontData"
require("Hmisc")
baseDir <- getwd()
#This will compare the average population fitness for a run of experiments

plotAverages <- function()
{
  files = list.files(path=baseDir, pattern = "Pop[0-9]+")
  noOfRuns <- length(files)
  displacement <- c()
  weight <- c()
  tolerance <-c()
  allDisp <- c()
  dispDevs <- c()
  allWeight <- c()
  allTol <-c()

  par(mfrow=c(3,3))
  genCount <- 1
  for(file in files)
    {
      print(file)
      title = paste("generation:",genCount,sep=" ")
      localFile = paste(baseDir,file, sep = "/")
      tmpExperiment <- read.table(localFile);
      
      displacement <- tmpExperiment$V1
      weight <- tmpExperiment$V2
      tolerance <- tmpExperiment$V3
      print(max(displacement))
      print(max(weight))

      genCount <-genCount+1
      aveDisp  <- mean(displacement)
      aveWeight  <- mean(weight)
      aveTol <- mean(tolerance)

      dispDev <- sd(displacement)

      allDisp <- c(allDisp,aveDisp)
      allWeight <- c(allWeight,aveWeight)      
      allTol <- c(allTol,aveTol)
      dispDevs <- c(dispDevs, dispDev)
    }
  
  postscript(file="averages.ps", paper="special",
  width=8, height=8, onefile=TRUE, encoding="TeXtext.enc",
  horizontal=FALSE)
  plot(c(1:49),allDisp, pch=16,lty=1,main="Displacement",
       xlab="generations",ylab="average displacement")
  lines(c(1:49),allDisp)
  ## for(i in seq(1,49,5))
  ##   {
  ##     errbar(i,allDisp[i],allDisp[i]+dispDevs[i],allDisp[i]-dispDevs[i],add="true")
  ##   }

  
  
  plot(c(1:49),allWeight, pch=16,lty=1,main="Weight",
       xlab="generations",ylab="average displacement")
  lines(c(1:49),allWeight)
  plot(c(1:49),allTol, pch=16,lty=1,main="Tolerance",
       xlab="generations",ylab="average displacement")
  lines(c(1:49),allTol)
  dev.off()
}
plotAverages()
