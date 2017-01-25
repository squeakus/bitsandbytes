require("gplots")
require("Hmisc")
require("matrixStats")

meanFolders <- function()
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
  meanLength <- rowMeans(lengthArray)
  meanStress <- rowMeans(stressArray)
  stddevLength <- rowSds(lengthArray)
  stddevStress <- rowSds(stressArray)
  matplot(meanLength, type='l')
  for(i in seq(5,length(meanLength),10))
    {
     errbar(i,meanLength[i],meanLength[i]+stddevLength[i],meanLength[i]-stddevLength[i],,,add="true")
    }
  X11()
  matplot(meanStress, type='l')
  for(i in seq(5,length(meanStress),10))
    {
     errbar(i,meanStress[i],meanStress[i]+stddevStress[i],meanStress[i]-stddevStress[i],,,add="true")
    }
}
meanFolders()

