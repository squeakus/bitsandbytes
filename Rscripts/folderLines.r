require("gplots")
require("Hmisc") # for error bars
# Generates a lineplot for each run in the folder

folderLines <- function(folder='none')
{
  genCol = 1 # index of generations
  resCol = 2 # results column
  title = "result"
  allNames <- c()
  # find current folder
  if(folder == 'none'){
      folder <- getwd()
    }
  print(sprintf("graphing folder:%s",folder))
  
  # define variables
  files = list.files(path=folder,pattern="*.dat")
  noOfRuns = length(files)
  fullName = paste(folder, files[1], sep='/')
  sampleIndiv <- read.table(fullName,sep=",");
  noOfGens <- length(sampleIndiv[,genCol])
  resultArray <- matrix(NaN, nrow=noOfGens,ncol=noOfRuns)

  # read the dat files
  run = 0
  for(file in files)
  {
    run <- run + 1
    fullName = paste(folder, file, sep='/')
    result <- read.table(fullName,sep=",")
    resultArray[,run] <- result[,resCol]
    allNames <- append(allNames,file)
  }
  
  # output to file
  postscript(file='result.ps', paper="special",width=8,
             height=8, onefile=FALSE, encoding="TeXtext.enc",
             horizontal=FALSE)
  
  #plot mean of the runs
  matplot(resultArray,type ="l", main=title,
          lty=1,xlab ="generation", ylab= "fitness")
  smartlegend(x="left", y="bottom",allNames,
              col=c(1:length(allNames)), lty=1, inset=0)
  dev.off()#close output file
}

