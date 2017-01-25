require("gplots")

graphColumns <- function(experiment)
  {
    baseDir <- "/Users/jonathanbyrne/results/toribash"
    file <- paste(baseDir,experiment, sep = "/")
    print(file)
    experimentArray <- read.table(file)
    matplot(experimentArray,type = "l",ylim = c(min(experimentArray),max(experimentArray)))
  }
