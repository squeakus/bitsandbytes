doubleMe x = x+x
doubleUs x y = doubleMe x + doubleMe y
doubleSmallNumber x = if x > 100 
		  then x 
		  else x*2
conan'Obrien = "its-a ME!"
boomBang xs = [if x < 10 then "BOOM" else "BANG" | x<-xs, odd x]
length' xs = sum [1| _ <-xs]
removeNonUpperCase st = [c | c <- st, c `elem` ['A'..'Z']]