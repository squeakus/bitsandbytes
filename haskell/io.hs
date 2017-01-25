import Data.Char  
{-
main = do  
    putStrLn "Hello, what's your name?"  
    name <- getLine  --have to assign as it is an IO String not just a string
    putStrLn ("Hey " ++ name ++ ", you rock!")
-}  
main = do  
    putStrLn "What's your first name?"  
    firstName <- getLine  
    putStrLn "What's your last name?"  
    lastName <- getLine  
    let bigFirstName = map toUpper firstName  
        bigLastName = map toUpper lastName  
    putStrLn $ "hey " ++ bigFirstName ++ " " ++ bigLastName ++ ", how are you?"