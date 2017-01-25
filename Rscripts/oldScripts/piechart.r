postscript(file='survey.ps', paper="special",
width=8, height=8, onefile=FALSE, encoding="TeXtext.enc",
horizontal=FALSE)

pie(c(55.9,36.84,7.26),main="Survey Results", labels=c("Low Fitness","High Fitness","No Preference"), col=rainbow(3))
