local turns = 0
local current_frame= 10

-- it seems the order of the functions is important :(
local function joint_random()
        for k,v in pairs(JOINTS) do
                set_joint_state(0, v, math.random(1,4))
        end
        set_grip_info(0, BODYPARTS.L_HAND, math.random(0,2))
        set_grip_info(0, BODYPARTS.R_HAND, math.random(0,2))
end


local function next_turn()
   if current_frame == 10 then
      joint_random()
      current_frame = 0
   end
   current_frame = current_frame + 1
   step_game()
end



local function next_game()
   tori_score = get_player_info(1).injury
   uke_score = get_player_info(0).injury
   turns = turns + 1
   echo("now in round "..turns.." tori:"..tori_score.." uke: "..uke_score)
   start_new_game()
end





local function start()
   next_turn()
end

add_hook("enter_freeze","keep stepping",next_turn)
add_hook("end_game","start next game",next_game)
add_hook("new_game","start",start)
