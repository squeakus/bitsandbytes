require("gplots")

bplot <-function(experimentName, resultsArray, range)
{
names(resultsArray) <- c("IntFlip","Structural","Nodal")  
fileName <- paste("/Users/jbyrne/Desktop/graphs/codonbplots/",experimentName,"Invalids",sep="")
fileName <- paste(fileName, "eps", sep=".")
postscript(file=fileName, paper="special", width=4, height=8, onefile=FALSE, encoding="TeXtext.enc", 		horizontal=FALSE)
barplot(resultsArray, width = .2, ylim =c(0,range),ylab = "fitness",col = c("grey", "grey", "grey"), main = experimentName)
dev.off()
}
