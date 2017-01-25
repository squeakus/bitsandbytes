require("gplots")
require("Hmisc")
#require("matrixStats")

plotFolders <- function()
{
  lengthArray <-NULL
  stressArray <-NULL

  files = list.files(pattern="*Euc")
  
  postscript(file="Euclid.ps", paper="special",
  width=11, height=8, onefile=TRUE, encoding="TeXtext.enc",
  horizontal=TRUE)
  par(mfrow=c(2,3))
  for(file in files)
    {
      info <- c(file.info(file))
      if(info$isdir == TRUE)
      {
        first = TRUE
        colCount = 0
        print(sprintf("folder: %s",file))
        folder <- paste(getwd(),file,sep='/')
        datFiles = list.files(path=folder,pattern="*.dat")
        noOfRuns <- length(datFiles)
        for(dat in datFiles)
           {
             fullName <- paste(folder,dat,sep='/')
             print(fullName)
             tmpTable <- read.table(fullName);
             xCoords = tmpTable$V1
             yCoords = tmpTable$V2
             if(first){
             plot(xCoords,yCoords,xlim=c(0,350),ylim=c(0,400),main=dat,type='l',col=colCount)
             first=FALSE}
             else{
               lines(xCoords,yCoords,col=colCount)
             }
             colCount = colCount+1
          }
      }
    }
  print("Finished")
  dev.off()
}
plotFolders()

