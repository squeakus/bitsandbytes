require("gplots")

majorTitles <- c("WordMatch", "EvenFiveParityFitnessBSF", "SantaFeAntTrailBSF", "SymbolicRegressionJScheme")
                                        # The names of the colour groups
minorTitles <- c("SPC", "CSC")

                                        # Root directory of experiment directories
baseDir <- "/Users/erikhemberg"
remote <- "/Volumes/erikhemberg"
cmd <- paste("mount | column -t | awk '{print$3}' | grep", remote)
if(system(cmd) == remote) {
  baseDir <- remote
}
experimentDir <- paste(baseDir, "Documents/projects/GE_Operators/Experiments/Runs/v2", sep="/")
                                        #
path2 <- "r2"
mutK <- c("m_cs_m", "m_std_m")
mutV <- c("0.01", "0.1", "0.9")
xoK <- c("xo_cs_xo", "xo_std_xo")
xoV <- c("0.1", "0.5", "0.9")
wm <- list(title = "WordMatch", path="r1_wm", baseId="r1_wm_wm")
efp <- list(title = "EvenFiveParity", path=path2, baseId="r2_efp")
sf <- list(title = "SantaFeAnt", path=path2, baseId="r2_sf")
sr <- list(title = "SymbolicRegression", path=path2, baseId="r2_sr")

experiments <- list(wm,efp,sf,sr)

variants <- array(NA,dim(length(mutK)*length(mutV)*length(xoK)*length(xoV)))
cnt <- 1
for(i in 1:length(mutK)) {
  for(j in 1:length(xoK)) {
    for(k in 1:length(mutV)) {
      for(l in 1:length(xoV)) {
        variants[cnt] <- paste(xoK[j],xoV[l],mutK[i],mutV[k],sep="_")
        cnt <- cnt + 1
      }
    }
  }
}
                                        #Padding between outputs
breakEvery <- 6
                                        #Colors to alternate between
colours <- c("grey", "brown")

totalTests <- 30

                                        #
                                        # Some output labels
                                        #
label1 <- "XO pr"
label2 <- c("(0.1", "0.5", "0.9)", "")
                                        #
                                        # Set how many times [label1] and [label2] should repeat
                                        #
labelRepeat <- 6

label3 <- c(        "IFMut (0.01)")
label3 <- c(label3, "CSMut (0.01)")
label3 <- c(label3, "IFMut (0.1)")
label3 <- c(label3, "CSMut (0.1)")
label3 <- c(label3, "IFMut (0.9)")
label3 <- c(label3, "CSMut (0.9)")

cnt <- 0
while(cnt < length(experiments))
  {
    cnt <- cnt + 1
    exp <- experiments[[cnt]]
                                        # Setup for looping through
                                        # all directories
    allBest <- NULL
    currentBreak <- breakEvery
    for(variant in variants) {
      v <- paste(exp$baseId, variant,sep="_")
      currentDir <- paste(experimentDir,exp$path,v,"",sep="/")
      files = list.files(path=currentDir, pattern=".dat")
      print(currentDir)
      
                                        # For each file in the current
                                        # directory, load the best
                                        # fitness from each run into a
                                        # vector
      thisBest <- NULL
      for(file in files)
        { filePath <- paste(currentDir, file, sep="/")
          fileData <- read.csv(file=filePath, head=TRUE, sep=" ")
          thisBest <- c(thisBest, fileData[dim(fileData)[1], 1])
        }

                                        # Pad the vector to fill
                                        # values for runs that crashed
      while(length(thisBest) < totalTests)
        { thisBest <- c(thisBest, NaN)
        }

                                        # Add the experiment to the
                                        # matrix of experiments
      allBest <- cbind(allBest, matrix(thisBest))
                                        # Put in a dummy
      currentBreak <- currentBreak - 1
      if(currentBreak <= 0)
        { allBest <- cbind(allBest, 1:length(thisBest)*NA)
          allBest <- cbind(allBest, 1:length(thisBest)*NA)
          currentBreak <- breakEvery
        }
    }
                                        # Render all the experiment
                                        # data

    allBest <- data.frame(allBest)
    fileName <- paste(experimentDir, majorTitles[cnt], sep="/")
    fileName <- paste(fileName, "eps", sep=".")
    postscript(file=fileName, paper="special", width=10, height=10, onefile=FALSE, encoding="TeXtext.enc", horizontal=FALSE)
    boxplot(allBest, col=colours, axes=FALSE)
    axis(2)
    mtext(side=1, line=-0.5, at=(1:(labelRepeat*4)*2-0.5), text=label2)
    mtext(side=1, line=0.5, at=(1:labelRepeat*(breakEvery+2)-4.5), text=label1)
    mtext(side=1, line=2, at=(1:length(label3)*(breakEvery+2)-4.5), text=label3)
    title(majorTitles[cnt], xlab="", ylab="bestFit")
    smartlegend(x="left", y="top", minorTitles, col=colours, lty=1:2, inset=0)
    dev.off()
  }
