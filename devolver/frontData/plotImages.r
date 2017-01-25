#This will compare the average population fitness for a run of experiments
require("hexbin")
require("gplots")

plotImages <- function()
{
  files = list.files(pattern = "Gen[0-9]+")
  noOfRuns <- length(files)
  xCoords <- c()
  yCoords <- c()
  genCount <- 1

  
  for(file in files)
    {    
      print(file)
      print(sprintf("%02d",genCount))
      filename = paste("plot",sprintf("%02d",genCount),sep=".")
      filename = paste(filename,"ps",sep=".")
      
      title = paste("generation:",genCount,sep=".")
      postscript(file=filename, paper="special",
                 width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
                 bg='white',horizontal=FALSE)
      tmpExperiment <- read.table(file);
      xCoords <- tmpExperiment$V1
      yCoords <- tmpExperiment$V2
      lowess(xCoords, yCoords, f=2/3, iter=3, delta=.01*diff(range(x)))
      par(usr=c(0,500,0,200))
      smoothScatter(xCoords, yCoords, xlim=c(0,500), ylim=c(0,200),
                    bandwidth=10,
                    pch=16, main=title, xlab="stresses(kN)",ylab="no. of beams")
      
      #hist2d(xCoords, yCoords, xlim=c(0,500), ylim=c(0,200), nbins =50,
      #       col=c("white", heat.colors(12)),main=title,xlab="stresses(kN)",ylab="no. of beams")
     
      genCount <-genCount+1
      dev.off()
    }
}
plotImages()
