require(simpleboot)
baseDir <- getwd()

plotLowess <- function()
{
  files = list.files(path=baseDir, pattern = ".dat")
  #par(mfrow=c(2,3))
  
  for(file in files){
    print(file)
    target <- read.table(file)
    name <- substr(file, 0, nchar(file)-12)
    filename = paste(name, 'ps', sep='.')

    postscript(file=filename, paper="special",width=8,
               height=8, onefile=TRUE, encoding="TeXtext.enc",
               horizontal=FALSE)

    
    plot(target, pch=19, main=name,xlab='Time (seconds)', ylab='Distance', ylim=c(0,600), xlim=c(0,120))
    #plot(target, pch=19, main=name)
    #lines(lowess(target), col=2)
    #print(loess(target))
    lo.b <- loess.boot(loess(target), R = 500, rows= FALSE)
    plot(lo.b)
    dev.off()
  }
}
plotLowess()
