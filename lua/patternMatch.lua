--http://www.wowwiki.com/API_gsub
testString = "<moo>!blah<moo>"
print(testString)
--add a space
--newString = string.gsub(testString,"<"," %1")
--add two spaces
--newString = string.gsub(testString,"<%a*>"," %1 ")
--only take chunk in brackets
newString = string.gsub(testString,"<(%a*)>","%1")


print(newString)
