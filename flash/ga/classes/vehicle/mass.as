class mass
{	
	var pos:cartesian;
	var old:cartesian;
	var vel:cartesian;
	var parent:vehicle;
	var radius:Number;
	var contact:Boolean;
	var collide:Boolean;
	var friction:Number;
	var grav:Boolean;
	
	function drive(pDir:cartesian)
	{
		pos.inc(pDir.factor(-pDir.dot(vel)*friction));
		contact=true;
	}
	
	function draw()
	{
		var p=pos.toScreen();
		plot.dircle(parent.mymc, p.x, p.y, radius*_root.cam.zoom, 0xFF0000);
	}
	
	function update()
	{	
		if(grav){vel.inc(_root.grav)};
		
		vel=vel.factor(0.99);
		
		pos.inc(vel);
		
		contact=false;
		if(collide){_root.sectors.collide(this);}
		
		vel=pos.sub(old);
		old.equ(pos);
	}
	
	function mass(pPos:cartesian, pParent:vehicle) 
	{
		pos=new cartesian(pPos.x, pPos.y);
		old=new cartesian(pPos.x, pPos.y);
		vel=new cartesian(0,0);
		parent=pParent;
		radius=10;
		friction=0.2;
		collide=true;
		grav=true;
	}
}