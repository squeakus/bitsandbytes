-- SPIN THE WHEEL by war_hero
echo("Spin the wheel with the mouse. When you are happy with the speed, press [L] to drop it. Press [K] to pause")

angle = 0.00004
lastAngle = .00004
angleVel = 0
radius = 150
wheelVel = {x=0, y=0}
windowW, windowH = get_window_size()
stageMult = {x=1, y=1}
stageCenter = {x=windowW/2, y=windowH/2}
wheel = {}
wheel = stageCenter
mouseDown = false
last = {x=0, y=0}
fall = false
friction = .6
bounce = -.3
oldTime = get_world_state().frame_tick

function wheelFall()
	if(fall == true) then
		if(wheel.y < windowH-radius-10) then
			wheelVel.y = wheelVel.y + .982
		end
		if(wheel.y > windowH-radius-10) then
			angleVel = angleVel*(1-(friction/30))
			wheelVel.y = wheelVel.y*bounce
			wheelVel.x = (angleVel/360)*(2*radius*3.14159)
		end
		if(get_world_state().frame_tick > oldTime + 500) then
			if(wheel.x < 5+radius or wheel.x > windowW-5-radius) then
				wheelVel.x = wheelVel.x*bounce
				wheelVel.y = (angleVel/360)*(2*radius*3.14159)*friction
				angleVel = 0-angleVel
				if(wheelVel.y > 0 and wheel.y > windowH-radius-10 and wheel.x > windowW-15-radius) then
					wheelVel.y =0-wheelVel.y
				end
				oldTime = get_world_state().frame_tick
			end
		end
		wheel.x = wheel.x + wheelVel.x
		wheel.y = wheel.y + wheelVel.y
	end
end

function draw_stage()
	wheelFall()
	if(angle > 360) then
		angle = angle - 360
		lastAngle = lastAngle - 360
	end
	if(angle < -360) then
		angle = angle + 360
		lastAngle = lastAngle + 360
	end
	if(mouseDown == false) then
		lastAngle = angle
		angle = angle + angleVel
	end
	draw_wheel(wheel.x, wheel.y, angle)
end

function mouseClicked()
	mouseDown = true
	angleVel = 0
end

function mouseUnClicked()
	mouseDown = false
end
	
function onMouseMove(mx, my)
	if(mouseDown == true) then
		p1 = {x=mx, y=my}
		p2 = last
		a = calculateDist(p1, p2)
		b = calculateDist(p1, wheel)
		c = calculateDist(p2, wheel)
		
		flip = lawCos(a, b, c)
		if(multAngles(p1, p2) > 0) then
			flip = flip*-1
		end
		lastAngle = angle
		angle = angle + flip
		angle = angleSmooth(angle, lastAngle, angleVel)
		angleVel = flip
		
	end
	last.x, last.y = mx, my
end	
	
function angleSmooth(ang, lastang, angvel)	
	z = (ang + (lastang + angvel))/2
	return z
end
	
function multAngles(p1, p2)
	edge = {x=-1000, y=0} 
	a = calculateDist(p1,edge )
	b = calculateDist(p1, wheel)
	c = calculateDist(edge, wheel)
	ang1 = lawCos(a, b, c)
	a = calculateDist(p2,edge )
	b = calculateDist(p2, wheel)
	c = calculateDist(edge, wheel)
	ang2 = lawCos(a, b, c)
	ang = ang1 - ang2
	if(ang>0 and p1.y < wheel.y) then
		return -1
	end
	if(ang<0 and p1.y < wheel.y) then
		return 1
	end
	if(ang>0 and p1.y > wheel.y) then
		return 1
	end
	if(ang<0 and p1.y > wheel.y) then
		return -1
	end
end	
	
function lawCos(a, b, c)
	nuAng = math.deg(math.acos(((b^2) + (c^2) - (a^2))/(2*b*c)))
	return nuAng
end	
	
function draw_wheel(px, py, angle)
	wheel.x = px
	wheel.y = py
	set_color(.5, .5, .5, 1)
	inner = 0
	outer = radius
	loops = 1
	start = 0
	slices = 30
	sweep = 360
	blend = 0
	draw_disk(wheel.x, wheel.y, inner, outer, slices, loops, start, sweep, blend)
	draw_bar(px, py, angle)
	
end	

function draw_bar(px, py, angle)
	p1 = {x=0, y=0}
	p1.x = px
	p1.y = py
	p2 = {x=0, y=0}
	p2.x = p1.x + radius*math.cos(math.rad(angle))
	p2.y = p1.y + radius*math.sin(math.rad(angle))
	set_color(0, 0, 0, 1)
	k = calculateDist(p1, p2)
	--echo("k: " .. k)
	for i=0, k do
		z = {x=0, y=0}
		z = calculateLine(p1, p2, i, k)
		draw_quad(z.x, z.y, 10, 10)
	end	
end

function calculateDist(p1, p2)
	z = math.sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
	return z
end

function calculateLine(p1, p2, i, k)
	z = {x=0, y=0}
	z.x = math.floor(((i/k)*p1.x) + (((k-i)/k)*p2.x))
	z.y = math.floor(((i/k)*p1.y) + (((k-i)/k)*p2.y))
	return z
end

function keyDown(key)
	if(key == 108) then
		fall = true
	end
	if(key == 107) then
		wheel.x = stageCenter.x
		wheel.y = stageCenter.y
		fall = false
	end
end

add_hook("draw2d", "stages", draw_stage)
add_hook("mouse_button_down","clicking",mouseClicked)
add_hook("mouse_button_up","clicking",mouseUnClicked)
add_hook("mouse_move","moving",onMouseMove)
add_hook("key_down","keys",keyDown)
