require("gplots")
require("Hmisc")
require("matrixStats")
#require("matrixStats")

#add error bars for single set of stdDev
addErrBars <- function(stdDev, mean, step=20){
  noOfGens <- 49
  print(sprintf("gens:%s",noOfGens))
  for(gen in seq(5,noOfGens,step)){
      if((mean[gen]-stdDev[gen]) >0){
            errbar(gen,mean[gen],
                   mean[gen]+stdDev[gen],
                   mean[gen]-stdDev[gen],
                   ,,add="true",lty=1)
      }
  }
}


plotFolders <- function()
{
  lengthArray <-NULL
  stressArray <-NULL
  
  files = list.files(pattern="run+")
  for(file in files)
    {
      info <- c(file.info(file))
      if(info$isdir == TRUE)
      {
        print(sprintf("folder: %s",file))
        folder <- paste(getwd(),file,sep='/')
        datFiles = list.files(path=folder,pattern="Pop[0-9]+.dat")
        noOfGens <- length(datFiles)
        avrLength <- NULL
        avrStress <- NULL
        for(dat in datFiles)
          {
            fullName <- paste(folder,dat,sep='/')
            tmpTable <- read.table(fullName);
            avrStress <- c(avrStress,mean(tmpTable$V1))
            avrLength <- c(avrLength,mean(tmpTable$V2))
          }
        lengthArray <-cbind(lengthArray,matrix(avrLength))
        stressArray <-cbind(stressArray,matrix(avrStress)) 
      }
    }
  lengthMean <- rowMeans(lengthArray)
  lengthDev <- rowSds(lengthArray)
  stressMean <- rowMeans(stressArray)
  stressDev <- rowSds(stressArray)

  
  postscript(file='result.ps', paper="special",width=16,
             height=8, onefile=TRUE, encoding="TeXtext.enc",
             horizontal=FALSE)
  par(mfrow=c(1,2))

  matplot(lengthArray, type='l', ylim=c(50,150))
  #X11()
  matplot(lengthMean, type='l', ylim=c(50,150))
  addErrBars(lengthDev, lengthMean, 10)

  #X11()
  matplot(stressArray, type='l', ylim=c(50,250))
  #X11()
  matplot(stressMean, type='l', ylim=c(50,250))
  addErrBars(stressDev, stressMean, 10)
  dev.off()
}
plotFolders()


