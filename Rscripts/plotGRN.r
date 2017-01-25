#!/usr/bin/Rscript

require("gplots")

filePath <- commandArgs()[6]
fileContents <- read.table(filePath, header=TRUE);
print(length(fileContents[1,]));
fileName <- paste(filePath, "ps", sep=".")
graphTitle <- filePath

lineTypes <- array(dim=c(length(colnames(fileContents))))
lineColours <- array(dim=c(length(colnames(fileContents))))

tfs <- grepl("TF", colnames(fileContents))
is <- grepl("I", colnames(fileContents))
ps <- grepl("P", colnames(fileContents))

lineTypes[tfs] <- 3
lineTypes[is] <- 2
lineTypes[ps] <- 1

lineColours[tfs] <- 1:length(lineColours[tfs])
lineColours[is] <- 1:length(lineColours[is])
lineColours[ps] <- 1:length(lineColours[ps])

postscript(file=fileName, onefile=TRUE, encoding="TeXtext", horizontal=TRUE, width=11, height=8)
matplot(fileContents,
        main=graphTitle,
        type="l",
        lty=lineTypes,
        col=lineColours,
        xlab="GRN Timestep",
        ylab="Protein Concentration",
        ylim=c(0, max(fileContents)),
        xlim=c(0, length(fileContents[,1])),
        xaxt = "n")
axis(1, at=seq(0, length(fileContents[,1]), length(fileContents[,1])/10), labels=seq(0, length(fileContents[,1]), length(fileContents[,1])/10))

smartlegend(x="right", y="top", colnames(fileContents), col=lineColours, lty=lineTypes, inset=0)

dev.off()
