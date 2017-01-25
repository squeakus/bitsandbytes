require(bootstrap)
require(simpleboot)
baseDir <- getwd()

plotLowess <- function()
{
  files = list.files(path=baseDir, pattern = "t*.dat")
  #par(mfrow=c(2,3))
  
  for(file in files){
    print(file)
    target <- read.table(file)
    name <- substr(file, 0, nchar(file)-12)
    filename = paste(name, 'ps', sep='.')

    postscript(file=filename, paper="special",width=6,
               height=6, onefile=TRUE, encoding="TeXtext.enc",
               horizontal=FALSE)

    
    plot(target, pch=19, main=name,xlab='Time (seconds)', ylab='Distance', ylim=c(0,600), xlim=c(0,120))
    
    #loObj <- loess(target)
    #going to bootstrap with replacement to see how accurate the loess is

    for(i in 1:50)
      {
        rowSample <- NULL
        rowCount = length(target$V1)

        #build a sample
        for(i in 1:rowCount)
          {
            index <- sample(1:rowCount,1,replace=TRUE)
            value <- target[index,]
            rowSample <- rbind(rowSample, value) 
          }
        lines(lowess(rowSample), col=3)
      }
    #show the original
    lines(lowess(target), col=2)
    
    dev.off()
  }
}
plotLowess()
