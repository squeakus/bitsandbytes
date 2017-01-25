

getFolders <- function(){
  files = list.files()
  baseDir <- getwd()
  for(file in files){
    fileInfo = c(file.info(file))
    if(fileInfo$isdir == TRUE){
      fullDir <- paste(baseDir, file, sep="/")
      #print(sprintf("Directory: %s",fullDir))
      folderMean(fullDir)
    }
  }
}
