require(Hmisc)



age <- rnorm(1000,50,10)
sex <- sample(c('female','male'),1000,TRUE)
print(split(age,sex))
out <- histbackback(split(age, sex), probability=TRUE, xlim=c(-.06,.06),
                    main = 'Nodal Mutation Fitness Change')


#! just adding color 
barplot(-out$left, col="red" , horiz=TRUE, space=0, add=TRUE, axes=FALSE)
barplot(out$right, col="blue", horiz=TRUE, space=0, add=TRUE, axes=FALSE)
