postscript(file='survey.ps', paper="special",
width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
horizontal=FALSE)

barplot(c(55.9,36.84,7.26),main="Survey Results",  
   ylab="Percentage", names.arg=c("Low Fitness","High Fitness","No Preference"), 
   border="black",ylim=c(0,100))
