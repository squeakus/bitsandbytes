{-
define what can be passed into the function
define the base condition
define patterns to check the input is correct
add your guards | if necessary
define the process to be carried out on the list plus the recursive part
where and let are opposite sides of the same coin
$ and . makes things right associative
-}


{-bog standard function-}
doubleMe x = x+x
{-2 variables-}
doubleUs x y = doubleMe x + doubleMe y
{-if statement-}
doubleSmallNumber x = if x > 100 
		  then x 
		  else x*2
{-defining a variable-}                       
conan'Obrien = "its-a ME!"

{-takes a list, check if index is odd, goes boom or bang-}
boomBang xs = [if x < 10 then "BOOM" else "BANG" | x<-xs, odd x]

{-recursive length function, has function definition-}
{-length' xs = sum [1| _ <-xs]-}
length' :: (Num b)=> [a] -> b
length' [] = 0
length' (_:xs) = 1 + length' xs

{-function definition examples::-}
removeNonUpperCase :: [Char] -> [Char]
removeNonUpperCase st = [c | c <- st, c `elem` ['A'..'Z']]

{-uses-}
factorial :: Integer -> Integer
factorial n = product [1..n]

fib :: Integer -> Integer
fib 0 = 0 
fib 1 = 1  
fib n = fib(n-1) + fib(n-2) 

circumference :: Double -> Double  
circumference r = 2 * pi * r

n :: Int
n = 6
x :: Float
x = fromIntegral n

sayMe ::(Integral a) => a -> String
sayMe 1 = "one!"
sayMe 2 = "two!"
sayMe 3 = "three!"
sayMe x = "no fookin idea!"

recFact ::(Integral a) => a -> a
recFact 0 = 1 
recFact x = x * recFact (x -1)

addVectors :: (Num a) => (a,a) -> (a,a) -> (a,a)
{-addVectors a b = ( fst a + fst b, snd a + snd b)-}
addVectors (x1,y1) (x2,y2) = (x1 + x2, y1 + y2)

first :: (a, b, c) -> a
first (x,_,_) = x 
  
second :: (a, b, c) -> b  
second (_, y, _) = y  
  
third :: (a, b, c) -> c  
third (_, _, z) = z 

xs = [(1,2),(2,3),(4,5),(5,6)]
{-xs = 1:2:3:[]-}

{-pattern matching for functions-}
head' :: [a] -> a
head' [] = error "oi! thats an empty list" 
head' (x:_) = x

tell :: (Show a ) => [a] -> String
tell [] = "the list is empty"
tell (x:[]) = "the list has one element: " ++ show x 
tell (x:y:[]) = "there are two elements: " ++ show x ++ " and " ++ show y
tell (x:y:_) = "Theres three or more elements: " ++ show x ++ " and " ++ show y

{--
sum' :: (Num a) => [a] -> a
sum' [] = 0
sum' (x:xs) = x + sum' xs 
--}

capital :: String -> String
capital "" = "Thats an empty string"
capital all@(x:xs) = "the first letter of " ++ all ++ " is  " ++ [x]

bmiTell :: (RealFloat a) => a -> a -> String
bmiTell weight height
  | bmi <= skinny = "your skinny"
  | bmi <= fine = "your fine"       
  | bmi <= fat = "your fat"
  | otherwise = "your really fat"
  where bmi = weight /height ^ 2
        (skinny, fine, fat) = (18, 25, 35)
max' :: (Ord a) => a -> a -> a  
max' a b | a > b = a | otherwise = b  

myCompare :: (Ord a) => a -> a -> Ordering  
a `myCompare` b  
    | a > b     = GT  
    | a == b    = EQ  
    | otherwise = LT 
                  
initials :: String -> String -> String
initials firstName secondName = [f] ++ "." ++ [l] ++ "."
   where (f:_) = firstName
         (l:_) = secondName
         
cylinder :: (RealFloat a) => a -> a -> a
cylinder r h =
  let sideArea = 2 * pi * r * h
      topArea =  pi * r ^ 2
  in
   sideArea + topArea * 2
         
{-guard example with where, let doesn't work on guards-}
bmiCalc :: (RealFloat a) => a -> a -> String         
bmiCalc weight height
  |  bmi <= skinny = "your skinny!"
  |  bmi <= normal = "your normal"
  |  bmi <= fat = "your fat"
  |  otherwise =  "dear god"
    where bmi = weight / height ^ 2
          skinny = 18
          normal = 25
          fat = 30
         
{-calculate arrays of bmis using where-}          
calcBmis :: (RealFloat a) => [(a,a)] -> [a]
calcBmis xs = [bmi w h | (w,h) <- xs]
     where bmi  weight height = weight / height ^2
         
{-only returns weight of fat people using let-}           
letCalcBmis :: (RealFloat a) => [(a,a)] -> [a]
letCalcBmis xs = [bmi | (w,h) <- xs, let bmi = w / h^2, bmi >=25]

--returns head of list using a case statement, like pattern matching
headCase :: [a] -> a
headCase xs = case xs of [] -> error "Oi! thats an empty list"
                         (x:_) -> x
                         
--case can be used anywhere, not just definitions
describeList :: [a] -> String
describeList xs = case xs of [] -> "Its an empty list"
                             [x] -> "its a singleton"
                             xs -> "its a longer list"
                             
maximum' :: (Ord a) => [a] -> a
maximum' [] = error " maximum of empty list "
maximum' [x] = x
maximum' (x:xs)
  |x > maxTail = x
  |otherwise = maxTail where maxTail = maximum' xs
                             
--Recursive stuff
{-
replicate' :: Int -> Int -> [Int]
replicate' 0 _ = []
replicate' times thing = thing : replicate' thing (times-1) 
-}
replicate' :: (Num i, Ord i) => i -> a -> [a]
replicate' n x 
  | n <= 0 = []
  | otherwise = x: replicate'(n-1) x
                
take' :: (Num i, Ord i) => i -> [a] -> [a]
take' n _
  | n <= 0 = []
take' _ [] = [] 
take' n (x:xs) = x: take' (n-1) xs    
  
reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = reverse xs ++ [x]

repeat' :: a-> [a]
repeat' x = x: repeat' x

zip' :: [a] -> [b] -> [(a,b)]
zip' [] _ = []
zip' _ [] = []
zip' (x:xs) (y:ys) = (x,y): zip' xs ys

{--
elem' :: (Eq a) => a -> [a] -> Bool
elem' a [] = False
elem' a (x:xs) 
  | a == x = True
  | otherwise = elem' a xs  
--}
                
quicksort :: ( Ord a ) => [a] -> [a]
quicksort [] = []
quicksort ( x : xs ) =
  let smallerSorted = quicksort [ a | a <- xs , a <= x ]
      biggerSorted = quicksort [ a | a <- xs , a > x ]
      {-
        smallerSorted = quicksort (filter (<=x) xs)  
        biggerSorted = quicksort (filter (>x) xs)
      -}
  in smallerSorted ++ [ x ] ++ biggerSorted

--Currying stuff!
multThree :: (Num a) => a -> a -> a -> a
multThree a b c = a * b * c

compareWithHundred :: (Num a, Ord a) => a -> Ordering
compareWithHundred x = compare 100 x

divideByTen :: (Floating a) => a -> a
divideByTen = (/10)

isAlphaNum :: Char -> Bool
isAlphaNum = (`elem`['A'..'Z'])

{-Higher Order Stuff-}
applyTwice :: (a -> a) -> a -> a
applyTwice f x = f(f x)

zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith' _ [] _ = []
zipWith' _ _ [] = []
zipWith' f (x:xs) (y:ys) = f x y : zipWith' f xs ys

flip' :: (a -> b -> c) -> b -> a -> c
flip' f x y = f y x

--map' :: (a -> b) -> [a] -> [b]
--map' _ [] = [] 
--map' f (x:xs) = f x : map f xs


multByTwo ::(Num a) => a -> a
multByTwo a = a*2

filter' :: (a -> Bool) -> [a] -> [a]
filter' _ [] = []
filter' p (x:xs)
  | p x       = x : filter' p xs 
  | otherwise = filter' p xs
  
largestDivisible :: (Integral a) => a -> a
largestDivisible target  = head (filter p [100000,99999..])
  where p x = x `mod` target == 0
        
{-
sum of all odds up to 10,000 using map, filter and takewhile
sum ( takeWhile ( < 10000) (filter odd (map (^2) [1..]))) 
using list comprehensions
sum ( takeWhile ( < 10000) [m | m <- [ n^2 | n <- [1..]], odd m])
-}

{-produces collatz sequence for a given number-}
chain :: (Integral a) => a -> [a]
chain 1 = [1]
chain n
  | even n = n : chain (n `div` 2)
  | odd n = n : chain (n*3 + 1)
            
numLongChains :: Int
numLongChains = length (filter isLong (map chain [1..100]))
  where isLong xs = length xs > 15
        
{--this makes a list of [(*1),(*2),etc] for the laugh   
let listOfFuns = map (*) [1..]
(listOfFuns !! 5) 5
--}

--lambdas!!!!!
numLongChains' :: Int
numLongChains' = length (filter (\xs -> length xs > 15) (map chain [1..100]))
  
--make functions on the fly!
--zipWith (\a b -> (a * 30 + 3) / b) [5,4,3,2,1] [1,2,3,4,5]

{- for fold you need to pass in an accumulator unless you use foldl1
foldl (+) 0 [1,2,3,4]
foldl (+) 3 [1,2,3,4]
foldl1 (+) [1,2,3,4,5]

sum' :: (Num a) => [a] -> a
sum' xs = foldl (\acc x -> acc + x) 0 xs
-}
sum' :: (Num a) => [a] -> a
sum' = foldl (+) 0    {-returns a function that takes a list-}

elem' :: (Eq a) => a -> [a] -> Bool
elem' y ys = foldl (\acc x -> if x == y then True else acc) False ys

map' :: (a -> b) -> [a] -> [b]
map' f xs = foldr (\x acc -> f x : acc) [] xs  {-the : here prepends to the array-}

foldMult ::(Num a) =>[a] -> a
foldMult xs = foldl (\acc x -> acc * x) 1 xs

maxFold :: (Ord a) => [a] -> a
maxFold = foldl1 (\acc x -> if x > acc then x else acc)

reverseFold :: [a] -> [a]
reverseFold = foldl (\acc x -> x : acc) []

filterFold :: (a -> Bool) -> [a] -> [a]
filterFold f = foldr (\x acc -> if f x then x : acc else acc) []

headFold :: [a] -> a
headFold  = foldr1 (\x _ -> x) {-kinda silly to do it this way-}


--scanl (+) 0 [1,2,3,4,5,67]
--scanl1 (\acc x -> if x > acc then x else acc)  [1,2,3,4,5,4,3,7,8,9]

sqrtSum :: Int
sqrtSum = length (takeWhile (< 1000) ( scanl1 (*) (map sqrt [1..]))) + 1

{- Instead of using lambda you can combine functions using .
map (\x -> negate (abs x)) [5,-3,-6,7,-3,2,-19,24]  
map (negate . abs) [5,-3,-6,7,-3,2,-19,24]  
map (negate.sum.tail) [[1..2],[2..5,[1..7]]
fn x = ceiling (negate (tan (cos (max 50 x))))
fn = ceiling . negate . tan . cos . max 50
-}

oddSquareSum :: Int
--oddSquareSum = sum (takeWhile (<10000) (filter odd (map (^2) [1..])))
oddSquareSum = sum . takeWhile (<1000) . filter odd . map (^2) $ [1..]
 
{-oddSquareSum =   ALWAYS TRY TO KEEP IT READABLE!
    let oddSquares = filter odd $ map (^2) [1..]  
        belowLimit = takeWhile (<10000) oddSquares  
    in  sum belowLimit  -}