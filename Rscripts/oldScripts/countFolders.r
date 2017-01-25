require("gplots")
require("Hmisc")
#require("matrixStats")
# This will take the experiments passed in and graph the mean of the best reults


countFolders <- function()
{
  files = list.files()
  for(file in files)
    {
      info <- c(file.info(file))
      if(info$isdir == TRUE)
      {
        print(file)
        print(info$size)
      }
    }
  #ncol(finf <- file.info(dir()))# at least six
  #print(finf['isDir'])
  ##for(file in finf)
  ##  {
  ##    print(file)
  ##    #print(file[isDir])
  ##  }
}
