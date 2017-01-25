library(rgl)


#mydata <- replicate(3, rnorm(200))
#print(mydata)
mydata <- read.table('random_points.dat')
colnames(mydata) <- c('x','y','z')

zeroone <- function(x) {
  x.r <- range(x, na.rm=TRUE)
  (x - x.r[1])/(x.r[2]-x.r[1])
}

h <- zeroone(mydata$z)
col <- hsv(h=h,s=1,v=1)

plot3d(mydata, xlim=c(-1,2), ylim=c(-1,2), zlim=c(0,2), col=col)

