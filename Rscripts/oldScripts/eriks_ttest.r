library(quantreg)

OutBase = "RepairRuns_SmallPop_Main/Pop20_Gen500_TTest/" 
problems = c("SanFe", "WMatch", "SymReg") 
variations = c("TruncRep_no_wrap", "TruncRep_3_wrap", "Subtree_CreateRand_Rep_no_wrap", "Subtree_RouletteValid_Rep_no_wrap", "Subtree_RandValid_Rep_no_wrap", "Subtree_CreateRand_Rep_3_wrap", "Subtree_RouletteValid_Rep_3_wrap", "Subtree_RandValid_Rep_3_wrap", "Reg_no_wrap", "Reg_3_wrap")
shortVars = c("Trunc-nW", "Trunc-3W", "cRand-nW", "roulVal-nW", "randVal-nW", "cRand-3W", "roulVal-3W", "randVal-3W", "reg-nW", "reg-3W")
varCopy = variations

# nrow is the number of generations
# ncol is the number of trials
gens <- 500
trials <- 100
checkGen <- 20

# Set up the dir. for the data files
dataDir <- "/Users/swafford/Documents/GEExperiments/RepairRuns_SmallPop_Main/Pop20_Gen500/RepairRuns_SmallPop"

for (prob in problems){
  varCnt <- 0
  ttest_results <- matrix(NaN, nrow=length(variations), ncol=length(variations))
  #print(ttest_results)

  for (var in variations){
    var1Vec <- c()
    varCnt <- varCnt + 1
    currDir <- dataDir
    probVar <- ""
    probVar <- paste(probVar, prob, var, sep="_")
    #print(probVar)
    currDir <- paste(currDir, probVar, sep="")
    #print(currDir)
       
    # Grab all the .dat files in the given dir.
    files = list.files(path=currDir, pattern=".dat")
    bestFit <- matrix(NaN,nrow=gens,ncol=trials)
    cnt <- 0

    for (file in files){
      cnt <- cnt + 1
      filePath <- paste(currDir, file, sep="/")
      fileData <- read.table(file=filePath, head=FALSE, sep="")
      
      if (var == "Reg_no_wrap" || var == "Reg_3_wrap"){
        bestFit[,cnt] <- fileData$V1[checkGen]
      }
      else{
        bestFit[,cnt] <- fileData$V2[checkGen]
      }
    }

    # Average all the generations
    bestFitMean1 <- mean(bestFit)
    
    for (varC in varCopy){
      currDir2 <- dataDir
      probVar2 <- ""
      probVar2 <- paste(probVar2, prob, varC, sep="_")
      currDir2 <- paste(currDir2, probVar2, sep="")
      newFile = list.files(path=currDir2, pattern=".dat")
      bestFit2 <- matrix(NaN,nrow=gens,ncol=trials)
      cnt2 <- 0
      
      for (file in newFile){
        cnt2 <- cnt2 + 1
        filePath2 <- paste(currDir2, file, sep="/")
        fileData2 <- read.table(file=filePath2, head=FALSE, sep="")
        
        if (varC == "Reg_no_wrap" || varC == "Reg_3_wrap"){
          bestFit2[,cnt2] <- fileData2$V1[checkGen]
        }
        else{
          bestFit2[,cnt2] <- fileData2$V2[checkGen]
        }
      }
      
      bestFitMean2 <- mean(bestFit2)
      meanDiff = bestFitMean1 - bestFitMean2

      #print(bestFit)
      #print(bestFit2)
      print("mean difference")
      print(meanDiff)
      test <- t.test(x = bestFit,
                     y = bestFit2,
                     mu = meanDiff,
                     paired = TRUE,
                     var.equal = FALSE,
                     conf.level = 0.95
                     )
      print('P Val')
      print(test["p.value"][[1]])
      var1Vec <- c(var1Vec, test["p.value"][[1]])
    }

    ttest_results[,varCnt] <- var1Vec
  }

  #print(ttest_results)
  #out = paste(OutBase, prob, ".txt", sep="")
  #write(ttest_results,
  #      out,
  #      ncolumns = length(variations) + 1)

  my_tab <- as.table(ttest_results)
  colnames(my_tab) <- shortVars
  rownames(my_tab) <- shortVars

  print(my_tab)
  
  latex.table(my_tab,
              title = paste(prob, "-Test", sep=""),
              file = paste(OutBase, prob, "-", checkGen, sep=""),
              append = FALSE,
              caption = paste(prob, "Pairwise T-Test at Generation ", checkGen, sep=" "),
              label = paste("tab:", prob, "-ttest", sep=""),
              rowlabel = ""
              )
}
