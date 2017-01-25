require("gplots")
require("vioplot")

violinPlot <- function(experimentArray)
{
    baseDir <- "/Users/jonathanbyrne/results/bigDesignTest/blender3d_hof.bnf"
    colArray <- matrix(,5000,0)
    
    cnt <- 0
    for(experiment in experimentArray)
    {
      
      file <- paste(baseDir,experiment, sep = "/")
      experimentArray <- read.table(file)
      
      colArray <- cbind(colArray,experimentArray$V2)
    }
#code for plotting gorgeous violin plots
    fullFileName = paste(baseDir, "bighoftreeEditGraph.eps",sep = "/")
    postscript(file=fullFileName, paper="special",
    width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
    horizontal=FALSE)
    vioplot(c(colArray[,1]),c(colArray[,2]),names =c("Nodal","Structural"),col = "tomato")
    title(main = "Tree Edit Distance between derivation trees", ylab = "Distance")
    dev.off()
}


