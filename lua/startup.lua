-- Startup script
local hooks = { }
local events = {
	"new_game",
	"new_mp_game",
	"enter_frame",
	"end_game",
	"leave_game",
	"enter_freeze",
	"exit_freeze",
	"key_up",
	"key_down",
	"mouse_button_up",
	"mouse_button_down",
	"mouse_move",
	"player_select",
	"joint_select",
	"body_select",
	"draw2d",
	"draw3d",
	"play",
	"camera",
	"console",
	"bout_mouse_down",
	"bout_mouse_up",
	"bout_mouse_over",
	"bout_mouse_outside",
	"spec_mouse_down",
	"spec_mouse_up",
	"spec_mouse_over",
	"spec_mouse_outside"
}


function call_hook(event, ...)
	for key,func in pairs(hooks[event]) do
		local retval = func(unpack(arg))
		if (retval ~= nil and retval ~= 0) then
			-- If a non-zero value was returned, we skip the rest of the hooks
			-- Note that the hooks are not called in any specific order
			return retval
		end
	end
end

-- set_name is the name of the set of callbacks belonging to a script
function add_hook(event, set_name, func)
	hooks[event][set_name] = func
end

function remove_hook(event, set_name)
	hooks[event][set_name] = nil
end

function remove_hooks(set_name)
	for i,event in ipairs(events) do
		if (hooks[event][set_name] ~= nil) then
			hooks[event][set_name] = nil
		end
	end
end

-- Example
--[[
add_hook("new_game", "script_name", some_script_on_key_press)
remove_hook("script_name", "new_game")
remove_hooks("script_name")
--]]


------- Quick Hack to register callbacks -------
for i,v in ipairs(events) do
	hooks[v] = { } -- Create the table
	set_event_hook(v, function(...) return call_hook(v, unpack(arg)) end) -- Tell our C++ code to call our main hook
end
------- End of quick hack ------

-- Replace some of the default functions --
if (startup == nil) then
	local old_dofile = dofile
	dofile = function (filename)
		return old_dofile("./data/script/" .. filename)
	end

	local old_loadfile = loadfile
	loadfile = function (filename)
		return old_loadfile("./data/script/" .. filename)
	end

	local old_io_open = io.open
	io.open = function (filename, mode)
		-- Simple check for valid access
		if string.find(filename, "../", 1, true) or string.find(filename, "..\\", 1, true) or string.find(filename, "startup.lua", 1, true) then
			return nil, "invalid access"
		end

		return old_io_open("./data/script/" .. filename, mode);
	end

	function read_replay(filename)
		return old_io_open("./replay/" .. filename, "r")
	end

	startup = true
end

