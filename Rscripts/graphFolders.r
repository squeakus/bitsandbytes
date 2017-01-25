require("gplots")
require("Hmisc") # for error bars
source('processFolder.r')

graphFolders <- function(){
  files = list.files()
  baseDir <- getwd()
  allMeans <- c()
  allStdDev <- c()
  allNames <- c()

  for(file in files){
    fileInfo = c(file.info(file))
    if(fileInfo$isdir == TRUE){
      
      fullDir <- paste(baseDir, file, sep="/")
      result <- processFolder(fullDir)
      allMeans <- cbind(allMeans,result$Mean)
      allStdDev <- cbind(allStdDev,result$StdDev)
      allNames <- append(allNames,file)
    }
  }
  print(length(allNames))

  postscript(file='folders.ps', paper="special",width=8,
             height=8, onefile=TRUE, encoding="TeXtext.enc",
             horizontal=FALSE)

  matplot(allMeans, type ="l", main= "all results",
              lty=1,xlab ="generation", ylab= "fitness")

  addErrBars(allStdDev, allMeans, 50)

  smartlegend(x="left", y="bottom", allNames,
               col=c(1:length(allNames)), lty=1, inset=0)
  dev.off()
}


# This function will graph multiple stdDevs for different runs on the
# same graph
addErrBars <- function(stdDev, mean, step=20){
  dimensions <- dim(stdDev)
  noOfGens <- dimensions[1]
  noOfRuns <- dimensions[2]
  print(sprintf("gens:%s  runs:%s",noOfGens,noOfRuns))
  for (run in c(1:noOfRuns)){
    for(gen in seq(5,noOfGens-noOfRuns,step)){ 
      if((mean[gen+run,run]-stdDev[gen+run,run]) >0){
            errbar(gen+run,mean[gen+run,run],
                   mean[gen+run,run]+stdDev[gen+run,run],
                   mean[gen+run,run]-stdDev[gen+run,run],
                   ,,add="true",lty=1)
      }
    }
  }    
}
graphFolders()
