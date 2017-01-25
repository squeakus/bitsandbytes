class mouseHandler
{
	var pos:cartesian;
	var real:cartesian;
	var grid:Boolean;
	var down:Boolean;
	var press:Boolean;
	var release:Boolean;
	var old:Object;
	
	
	function onMouseDown()
	{
		down=true;
	}
	
	function onMouseUp()
	{
		down=false;
	}
	
	function onMouseWheel(delta)
	{
		_root.cam.tzoom+=delta/8;
		if(_root.cam.tzoom<0.2){_root.cam.tzoom=0.2}
		if(_root.cam.tzoom>2){_root.cam.tzoom=2}
	}
	
	function update()
	{
		old.pos=new cartesian(pos.x, pos.y);
		
		pos.x=_root._xmouse;
		pos.y=_root._ymouse;
		real=pos.toReal();
		
		if(grid){real.x=Math.round(real.x/10)*10; real.y=Math.round(real.y/10)*10;}
		
		if(!old.down && down){press=true}
		if(old.down && !down){release=true}
		if(old.press){press=false}
		if(old.release){release=false}
		
		old.down=down;
		old.press=press;
		old.release=release;
	}
	
	function mouseHandler() 
	{
		pos = new cartesian(0,0);
		real = new cartesian(0,0);
		old = new Object();
		grid=false;
		Mouse.addListener(this);
	}
}