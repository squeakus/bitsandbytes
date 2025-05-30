--[[
Author: Fraser McCrossan
Tested on G9, should work on most cameras.

An accurate intervalometer script, with pre-focus and screen power off options.

Features:
 - input is frame interval plus total desired run-time (or "endless")
 - displays frame count, frame total and remaining time after each frame
   (in endless mode, displays frame count and elapsed time)
 - honours the "Display" button during frame delays (so you can
   get it running then turn off the display to save power)
 - can turn off the display a given number of frames after starting
   (might take a couple of frames longer to cycle to correct mode)
 - can pre-focus before starting then go to manual focus mode
 - use SET button to exit 

 See bottom of script for main loop.
]]

--[[
@title Time-lapse
@param s Secs/frame
@default s 30
@param h Sequence hours
@default h 0
@param m Sequence minutes
@default m 1
@param e Endless? 0=No 1=Yes
@default e 0
@param f Focus: 0=Every 1=Start
@default f 0
@param d Display off frame 0=never
@default d 3
--]]

-- convert parameters into readable variable names
secs_frame, hours, minutes, endless, focus_at_start, display_off_frame = s, h, m, (e > 0), (f > 0), d

props = require "propcase"

-- derive actual running parameters from the more human-friendly input
-- parameters
function calculate_parameters (seconds_per_frame, hours, minutes, start_ticks)
   local ticks_per_frame = 1000 * secs_frame -- ticks per frame
   local total_frames = (hours * 3600 + minutes * 60) / secs_frame -- total frames
   local end_ticks = start_ticks + total_frames * ticks_per_frame -- ticks at end of sequence
   return ticks_per_frame, total_frames, end_ticks
end

function print_status (frame, total_frames, ticks_per_frame, end_ticks, endless)
   local free = get_jpg_count()
   if endless then
      local h, m, s = ticks_to_hms(frame * ticks_per_frame)
      print("#" .. frame .. ", " .. h .. "h " .. m .. "m " .. s .. "s")
   else
      local h, m, s = ticks_to_hms(end_ticks - get_tick_count())
      print(frame .. "/" .. total_frames .. ", " .. h .. "h" .. m .. "m" .. s .. "s/" .. free .. " left")
   end
end

function ticks_to_hms (ticks)
   local secs = (ticks + 500) / 1000 -- round to nearest seconds
   local s = secs % 60
   secs = secs / 60
   local m = secs % 60
   local h = secs / 60
   return h, m, s
end

-- sleep, but using wait_click(); return true if a key was pressed, else false
function next_frame_sleep (frame, start_ticks, ticks_per_frame)
   -- this calculates the number of ticks between now and the time of
   -- the next frame
   local sleep_time = (start_ticks + frame * ticks_per_frame) - get_tick_count()
   if sleep_time < 1 then
      sleep_time = 1
   end
   wait_click(sleep_time)
   return not is_key("no_key")
end

-- delay for the appropriate amount of time, but respond to
-- the display key (allows turning off display to save power)
-- return true if we should exit, else false
function frame_delay (frame, start_ticks, ticks_per_frame)
   -- this returns true while a key has been pressed, and false if
   -- none
   while next_frame_sleep (frame, start_ticks, ticks_per_frame) do
      -- honour the display button
      if is_key("display") then
	 click("display")
      end
      -- if set key is pressed, indicate that we should stop
      if is_key("set") then
	 return true
      end
   end
   return false
end

-- if the display mode is not the passed mode, click display and return true
-- otherwise return false
function seek_display_mode(mode)
   if get_prop(props.DISPLAY_MODE) == mode then
      return false
   else
      click "display"
      return true
   end
end

-- switch to autofocus mode, pre-focus, then go to manual focus mode
function pre_focus()
   local focused = false
   local try = 1
   while not focused and try <= 5 do
      print("Pre-focus attempt " .. try)
      press("shoot_half")
      sleep(2000)
      if get_prop(67) > 0 then
	 focused = true
	 set_aflock(1)
      end
      release("shoot_half")
      sleep(500)
      try = try + 1
   end
   return focused
end

if focus_at_start then
   if not pre_focus() then
      print "Unable to reach pre-focus"
   end
end

start_ticks = get_tick_count()

ticks_per_frame, total_frames, end_ticks = calculate_parameters(secs_frame, hours, minutes, start_ticks)

frame = 1
original_display_mode = get_prop(props.DISPLAY_MODE)
target_display_mode = 2 -- off

print "Press SET to exit"

while endless or frame <= total_frames do
   print_status(frame, total_frames, ticks_per_frame, end_ticks, endless)
   if display_off_frame > 0 and frame >= display_off_frame then
      seek_display_mode(target_display_mode)
   end
   shoot()
   if frame_delay(frame, start_ticks, ticks_per_frame) then
      print "User quit"
      break
   end
   frame = frame + 1
end

-- restore display mode
if display_off_frame > 0 then
   while seek_display_mode(original_display_mode) do
      sleep(1000)
   end
end

-- restore focus mode
set_aflock(0)

