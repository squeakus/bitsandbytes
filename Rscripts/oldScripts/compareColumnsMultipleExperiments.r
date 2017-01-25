require("gplots")
require("vioplot")

#this will compare multiple experiments and can compare stdDev, t-test, etc

compareColumns <- function(experimentArray)
{
    baseDir <- "/Users/jonathanbyrne/results/bigDesignTest/blender3d_hof.bnf"
    colArray <- matrix(,5000,0)
    
    cnt <- 0
    for(experiment in experimentArray)
    {
      
      file <- paste(baseDir,experiment, sep = "/")
      experimentArray <- read.table(file)
      
      colArray <- cbind(colArray,experimentArray$V1)
    }
#code for t-test
#    t.test(colArray[,1],colArray[,2],paired = FALSE,var.equal=FALSE)

#code for mean and std Dev    
#    mean <- colMeans(colArray) 
#    for(i in 1:length(experimentArray))
#    StdDev <- sd(colArray[i,])
#    print(sprintf("%.2f & %.2f",mean,StdDev))

#code for plotting columns
#    matplot(colArray,type = "l",ylim = c(0,max(colArray)), lty=c(1:2), col = 1)
#    smartlegend(x="right", y="top",c("nodal","structural"),col= 1,lty=c(1:2), inset=0)

# boxplot code
#    boxplot(colArray)
    
#code for plotting histograms  
   postscript(file="/Users/jonathanbyrne/ncdHist.eps", paper="special",
    width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
    horizontal=FALSE)

    hist(colArray, main = "Histogram for structural NCD", xlab = "distance")
    dev.off()
}
