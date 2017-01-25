

experiments <- c('Even Five', 'Santa Fe', 'Symbolic Reg', 'Word Match')
results <- c(10,20,50,20)
barplot(results, main="Problem Fitness Change", horiz=TRUE,
  names.arg=experiments)
