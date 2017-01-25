class bresenham
{
	static function getSectors(p1:cartesian, p2:cartesian, sectorSize:Number):Array
	{
		var sectors = new Array();
		
		var p=new cartesian(p1.x,p1.y);
		var grad=(p2.y-p1.y)/(p2.x-p1.x);
		var dir=new cartesian(p1.x<p2.x ? 1 : -1, p1.y<p2.y ? 1 : -1);
		
		var round=Math.round;
		var floor=Math.floor;
		var ceil=Math.ceil;
		var pow=Math.pow;

		
		var quick=true;
		if(floor((p.x-1)/sectorSize)!=floor(p2.x/sectorSize)){quick=false}
		if(floor((p.y-1)/sectorSize)!=floor(p2.y/sectorSize)){quick=false}
		
		var count=0;
		
		sectors.push(p1);
		
		do {	
			var xOK = floor(p.x/sectorSize)==floor(p2.x/sectorSize);
			var yOK = floor(p.y/sectorSize)==floor(p2.y/sectorSize);
			if((xOK && yOK)) {break}
			
			var a=new cartesian();
			a.x=round(floor(p.x/sectorSize+dir.x)*sectorSize);
			if(dir.x<0){a.x=round(ceil((p.x+1)/sectorSize+dir.x)*sectorSize)-1}
			a.y=round(p1.y+(a.x-p1.x)*grad);
			var b=new cartesian();
			b.y=round(floor(p.y/sectorSize+dir.y)*sectorSize);
			if(dir.y<0){b.y=round(ceil((p.y+1)/sectorSize+dir.y)*sectorSize)-1}
			b.x=round(p1.x+(b.y-p1.y)/grad);
			
			if(pow(a.x-p1.x, 2)+pow(a.y-p1.y, 2)<pow(b.x-p1.x, 2)+pow(b.y-p1.y, 2))
			{
				sectors.push(p=a);
			} else
			{
				sectors.push(p=b);
			}
		} while(count++<5000);
		
		return sectors;
	}
	function bresenham() 
	{
	}
}