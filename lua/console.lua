-- Variable Change Tester
-- USAGE: /echo a=b; c = a; b   = -34.34323   ; 
add_hook("console","",
    function(s,i)
        if(s:find("=")) then -- if the string incoming is a set variable command.
            if(s:sub(-1) ~= ";") then -- add ; to the end if it doesn't have it.
                s = s .. ";"
            end
            s = string.gsub(s,"%s","") -- Get Rid of spaces.
            for k, v in string.gmatch(s, "(.-)=(.-)%;") do -- Go through each "k=v;".
                _G[k] = v -- Set the variable k to v
                echo("Set: \"" .. k .. "\" to \"" .. v .. "\"") -- Echo to confirm change.
            end
            return 1 -- Don't show the line in chat.
        end
    end
)
