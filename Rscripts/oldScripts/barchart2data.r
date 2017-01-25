Frequency=c(3, 22, 18, 6, 11, 7, 6, 7, 6, 3, 2, 1, 0, 2, 1, 0, 0, 0, 0, 0)
mutType = rep(c("Nodal", "Structural"), 10)
print(Gender)
selections=rep(c(1,2,3,4,5,6,7,8,9,10), each=2)

postscript(file="click10c.ps", paper="special",width=6,
           height=6, onefile=TRUE, encoding="TeXtext.enc",
           horizontal=FALSE)

barplot(tapply(Frequency, list(mutType, selections), sum), col=c(2,4), legend=TRUE, beside=TRUE, ylim=c(0,25)) 
title(ylab="Number of Occurences", xlab="Number of Selections")

dev.off()
