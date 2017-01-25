rem Author: MLuna - based om MX3 sample script
rem Tested on S3IS only
rem Requires Fingalo's build v 119
rem Use with caution!

@title Motion Detection

rem Shot without auto-focus/with auto-focus/continuously (need to put in continuous mode manually)
rem T implies test mode with MD cells drawing and no shots taken
@param a Shot (0=nf/1=f/2=c/3=t)
@default a 1

rem How long the shutter button will be pressed in continuous mode
@param b Continuos shoot (secs)
@default b 10

@param c Threshold (0-255)
@default c 5

@param d Compare Interval (msecs)
@default d 20

@param e Compare Interval (secs)
@default e 0

rem If this value is too small, the camera goes continuously shooting after the 1st shot.
rem Experiment with this value to find one fitted to your needs
@param f Begin Delay (secs)
@default f 5

@param g Pix step(speed/accuracy adj)
@default g 5

@param h Columns 
@default h 6

@param i Rows 
@default i 6

rem Frame width in which no MD is performed (in cell units)
@param j Dead frame 
@default j 0

if a<0 then let a=0
if a>3 then let a=3
if c<0 then let c=0
if d<0 then let d=0
if e<0 then let e=0
if g<1 then let g=1
if h<1 then let h=1
if i<1 then let i=1
if j<0 then let j=0

rem Conversions secs to msecs
let b=b*1000
let e=e*1000
let f=f*1000

let d=d+e

rem This is the timeout in msecs. After this period, the motion trap is rearmed.
let T=600000

rem Parameters for the Dead Frame
let J=j+1
let H=h-j
let I=i-j

let t=0

print "press Shutter Button to Stop"

:repete

	md_detect_motion h, i, 1, T, d, c, 1, t, 1, J, J, H, I, 0, g, f

	if a=0 and t>0 then click "shoot_full"
	if a=1 and t>0 then shoot
	if a=2 and t>0 then goto "continuos"
	if a=3 then goto "test"

	let t=0

goto "repete"

:continuos
	let X=get_tick_count
	press "shoot_full"

	:contloop
		let U=get_tick_count
		let V=(U-X)
	if V<b then goto "contloop"

	release "shoot_full"
goto "repete"

:test
	if t>0 then print "Detected cells: ",t else print "No detection in 10 min!"
	let t=0
goto "repete"

end

