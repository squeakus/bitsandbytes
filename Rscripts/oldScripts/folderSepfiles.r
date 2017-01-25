require("gplots")
require("Hmisc")
#require("matrixStats")

plotFolders <- function()
{
  lengthArray <-NULL
  stressArray <-NULL

  files = list.files(pattern="*Tree")
 
  for(file in files)
    {
      plotName = paste(file,"ps",sep=".")
      print(plotName)
      postscript(file=plotName, paper="special",
                 width=8, height=8, onefile=FALSE,horizontal=FALSE, encoding="TeXtext.enc")
      
      info <- c(file.info(file))
      if(info$isdir == TRUE)
      {
        first = TRUE
        colCount = 0
        #print(sprintf("folder: %s",file))
        folder <- paste(getwd(),file,sep='/')
        datFiles = list.files(path=folder,pattern="*.dat")
        noOfRuns <- length(datFiles)
        for(dat in datFiles)
           {  
             fullName <- paste(folder,dat,sep='/')
             tmpTable <- read.table(fullName);
             xCoords = tmpTable$V1
             yCoords = tmpTable$V2
             if(first){
             plot(xCoords,yCoords,xlim=c(0,350),ylim=c(0,100),main=file,type='l',col=colCount,xlab="time(seconds)",ylab="distance")
             first=FALSE}
             else{
               lines(xCoords,yCoords,col=colCount)
             }
             colCount = colCount+1
          }
      }
    dev.off()
    }
  print("Finished!")
  
}
plotFolders()


