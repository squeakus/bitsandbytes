class wheel extends mass
{	
	var motor:Number;
	var angle:Number;
	var speed:Number;
	
	function drive(pDir:cartesian)
	{
		pos.inc(pDir.factor(motor*parent.dir));
		speed=pDir.dot(vel)/radius;
		angle+=speed;
		contact=true;
	}
	
	function draw()
	{
		var p=pos.toScreen();
		plot.wheel(parent.mymc, p.x, p.y, radius*_root.cam.zoom, 0x0000FF, angle);
	}
	
	function wheel(pPos:cartesian, pParent:vehicle) 
	{
		pos=new cartesian(pPos.x, pPos.y);
		old=new cartesian(pPos.x, pPos.y);
		vel=new cartesian(0,0);
		parent=pParent;
		radius=10;
		motor=0;
		angle=0;
		speed=0;
	}
}