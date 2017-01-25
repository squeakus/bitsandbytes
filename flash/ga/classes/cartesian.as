class cartesian
{
	var x,y:Number;
	
	function toScreen():cartesian
	{
		return new cartesian((x-_root.cam.pos.x)*_root.cam.zoom+250, (y-_root.cam.pos.y)*_root.cam.zoom+200);
	}
	
	function toReal():cartesian
	{
		return new cartesian((x-250)/_root.cam.zoom+_root.cam.pos.x, (y-200)/_root.cam.zoom+_root.cam.pos.y);
	}
	
	function equ(p:cartesian)
	{
		x=p.x;
		y=p.y;
	}
	
	function add(p:cartesian):cartesian
	{
		return new cartesian(x+p.x, y+p.y);
	}
	
	function sub(p:cartesian):cartesian
	{
		return new cartesian(x-p.x, y-p.y);
	}
	
	function inc(p:cartesian)
	{
		x+=p.x;
		y+=p.y;
	}
	
	function factor(p:Number):cartesian
	{
		return new cartesian(x*p, y*p);
	}
	
	function dot(p:cartesian):Number 
	{
		return x*p.x+y*p.y;
	}
	
	function length():Number
	{
		return Math.sqrt(Math.pow(x,2)+Math.pow(y,2));
	}
	
	function lenSqr():Number
	{
		return Math.pow(x,2)+Math.pow(y,2);
	}
	
	function cartesian(pX:Number, pY:Number)
	{
		x=pX;
		y=pY;
	}
}