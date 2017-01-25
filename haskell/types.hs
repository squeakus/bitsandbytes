{-learn about maybe and just-}

data Point = Point Float Float deriving (Show)
data Shape = Circle Point Float | Rectangle Point Point deriving (Show)

nudge :: Shape -> Float -> Float -> Shape
nudge (Circle (Point x y) r) a b = Circle (Point (x+a) (y+b)) r
nudge (Rectangle (Point x1 y1) (Point x2 y2)) a b = Rectangle (Point (x1 + a) (y1 + b)) (Point (x2 + a) (y2 + b))

{-first last age height ice-cream-}
{-
data Person = Person String String Int Float String deriving (Show)

firstName :: Person -> String
firstName (Person firstName _ _ _ _) =  firstName

lastName :: Person -> String
lastName (Person _ lastName _ _ _) =  lastName

age :: Person -> Int
age (Person _ _ age _ _) =  age

A better way of doing it
-}

data Person = Person { firstName :: String
                     , lastName :: String
                     , age :: Int    
                     , height :: Float
                     , flavor :: String
                     } deriving (Show)
                       
data Car = Car {company :: String, model :: String, year :: Int} deriving (Show)
{-Car {company="Ford", model="Mustang", year=1967}-}
  
tellCar :: Car -> String  
tellCar (Car {company = c, model = m, year = y}) = "This " ++ c ++ " " ++ m ++ " was made in " ++ show y  


{-Type synonyms just give types different names-}
type PhoneNumber = String  
type Name = String  
type PhoneBook = [(Name,PhoneNumber)] 

phoneBook :: PhoneBook  
phoneBook =      
    [("betty","555-2938")     
    ,("bonnie","452-2928")     
    ,("patsy","493-2928")     
    ,("lucille","205-2928")     
    ,("wendy","939-8282")     
    ,("penny","853-2492")     
    ]


inPhoneBook :: Name -> PhoneNumber -> PhoneBook -> Bool
inPhoneBook name pNumber pBook = (name, pNumber) `elem` pBook