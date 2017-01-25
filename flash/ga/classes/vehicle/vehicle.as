class vehicle
{	
	var masses:Array;
	var springs:Array;
	var dir:Number=1;
	var mymc:MovieClip;
	var best:Number;
	
	var head:mass;
	var rearWheel, frontWheel : wheel;
		
	function kill()
	{
		mymc.removeMovieClip();
	}
	
	function draw()
	{
		mymc.clear();
		for(var i in masses){masses[i].draw()}
		for(var i in springs){springs[i].draw()}
	}
	
	function update()
	{
		rearWheel.motor+=(1*0.5-rearWheel.motor)/10;
		for(var i in springs) {springs[i].update()}
		for(var i in masses) {masses[i].update()}
		
		rearWheel.motor+=(1*0.5-rearWheel.motor)/10;
		for(var i in springs) {springs[i].update()}
		for(var i in masses) {masses[i].update()}
		
		
		for(var i in masses) 
		{
			if (masses[i].pos.x>best) {best=masses[i].pos.x}
		}
		
		if(masses[0].contact || masses[1].contact)
		{
			_root.ga.score('crashed');
		}
	}
	
	function vehicle(p:dna) 
	{
		//trace(p.id());
		mymc=_root.createEmptyMovieClip("vehiclemc", 3);
		masses=new Array();
		springs=new Array();
		best=0;
		
		masses.push(new mass(new cartesian(p.px[0],-10-p.py[0]), this));
		masses.push(new mass(new cartesian(p.px[1],-10-p.py[1]), this));
		masses.push(new wheel(new cartesian(p.px[2],-10-p.py[2]), this));
		masses.push(new wheel(new cartesian(p.px[3],-10-p.py[3]), this));
		
		springs.push(new spring(masses[0], masses[1], this));
		springs.push(new spring(masses[0], masses[2], this));
		springs.push(new spring(masses[1], masses[3], this));
		springs.push(new spring(masses[0], masses[3], this));
		
		springs.push(new spring(masses[1], masses[2], this));
		springs.push(new spring(masses[2], masses[3], this))
		
		head=masses[0];
		rearWheel=masses[2];
		frontWheel=masses[3];
		
		for(var i in masses){masses[i].radius=p.radius[i]}
				
		for(var i in springs)
		{
			springs[i].springConstant=p.springConstant;
			springs[i].dampConstant=p.dampConstant;
		}
	}
}