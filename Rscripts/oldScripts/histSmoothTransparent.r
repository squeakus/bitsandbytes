require(ggplot2)

carrots <- data.frame(length = rnorm(100000, 6, 2))
cukes <- data.frame(length = rnorm(50000, 7, 2.5))

carrots$veg <- 'carrot'
cukes$veg <- 'cuke'

postscript(file='hist.ps', paper="special",width=8,
           height=8, onefile=TRUE, encoding="TeXtext.enc",
           horizontal=FALSE)


vegLengths <- rbind(carrots, cukes)
plot(c(0,0), pch=19,xlab='Time (seconds)',
     ylab='Distance', col= 'white')


p <- ggplot(vegLengths, aes(length, fill = veg)) + geom_density(alpha = 0.2)
ggsave(p, filename = "moo.eps", width=6, height=6)
dev.off()
