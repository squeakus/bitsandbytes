import System.IO  
import Data.Char  

main = do  
    handle <- openFile "haiku.txt" ReadMode  
    contents <- hGetContents handle  --gets contents from a handle
    putStr contents  
    hClose handle  
    contents <- readFile "haiku.txt"     
    writeFile "girlfriendcaps.txt" (map toUpper contents)
    
    