require(simpleboot)
set.seed(1234)

x <- runif(100)

## Simple sine function simulation
y <- sin(2*pi*x) + .2 * rnorm(100)
plot(x, y)  ## Sine function with noise
lines(lowess(x,y), col=2)

lo <- loess(y ~ x)

## Bootstrap with resampling of rows
lo.b <- loess.boot(lo, R = 50)

## Plot original fit with +/- 2 std. errors
plot(lo.b)

## Plot all loess bootstrap fits
plot(lo.b, all.lines = TRUE)

## Bootstrap with resampling residuals
lo.b2 <- loess.boot(lo, R = 50, rows = FALSE)
plot(lo.b2)
