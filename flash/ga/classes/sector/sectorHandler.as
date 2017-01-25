class sectorHandler
{
	var mymc:MovieClip;
	var sectors:Array;
	var lines:Array;
	var sectorSize:Number; 
	var onScreen:Array;
	
	function kill()
	{
		mymc.removeMovieClip();
	}
	
	function addRef(pX:Number, pY:Number, pObj:line)
	{
		if(pObj.length<3){return}
		var i=Math.floor(pX/sectorSize);
		var j=Math.floor(pY/sectorSize);
		if(sectors[i]==undefined){sectors[i]=new Array();}
		if(sectors[i][j]==undefined){sectors[i][j]=new sector(i,j);}
		sectors[i][j].addLine(pObj);
	}
	
	function newLine(pX1, pY1, pX2, pY2) //***NOTE: change these to cartesians***
	{
		var lineRef=new line(pX1, pY1, pX2, pY2);
		if(!lineRef.remove)
		{
			var hitSectors=bresenham.getSectors(new cartesian(pX1, pY1), new cartesian(pX2, pY2), sectorSize);
			for (var i in hitSectors) {addRef(hitSectors[i].x, hitSectors[i].y, lineRef,1)}
			lines.push(lineRef);
		}
	}
	
	
	function collide(pObj:Object)
	{
		var x = Math.floor((pObj.pos.x)/sectorSize-0.5);
		var y = Math.floor((pObj.pos.y)/sectorSize-0.5);
		
		sectors[x][y].resetCollided();
		sectors[x+1][y].resetCollided();
		sectors[x+1][y+1].resetCollided();
		sectors[x][y+1].resetCollided();
		
		sectors[x][y].collide(pObj);
		sectors[x+1][y].collide(pObj);
		sectors[x+1][y+1].collide(pObj);
		sectors[x][y+1].collide(pObj);
	}
	
	function draw() 
	{
		mymc.clear();
		var min=new cartesian(0,0);
		min=min.toReal();
		var max=new cartesian(500,400);
		max=max.toReal();
		min.x=Math.floor(min.x/sectorSize);
		min.y=Math.floor(min.y/sectorSize);
		max.x=Math.floor(max.x/sectorSize);
		max.y=Math.floor(max.y/sectorSize);
		
		onScreen=new Array();
		for(var i=min.x; i<=max.x; i++)
		{
			for(var j=min.y; j<=max.y; j++)
			{
				onScreen.push(sectors[i][j]);
			}
		}
		for(var i in onScreen){onScreen[i].resetDrawn(mymc)}
		for(var i in onScreen){onScreen[i].draw(mymc)}
	}
	
	function remove(p1, p2)
	{
		if(p2==undefined){p2=new cartesian(p1.x, p1.y)}
		var l=0;
		while(l<lines.length)
		{
			if(lines[l].remove)	{lines.splice(l,1)}
			l++;
		}
		
		var hitSectors=bresenham.getSectors(p1,p2,sectorSize);
		for (var s in hitSectors) 
		{
			var i=Math.floor(hitSectors[s].x/sectorSize);
			var j=Math.floor(hitSectors[s].y/sectorSize);
			sectors[i][j].remove();
		}
	}
	
	function erase(p:cartesian)
	{
		var x = Math.floor((p.x)/sectorSize-0.5);
		var y = Math.floor((p.y)/sectorSize-0.5);
		
		sectors[x][y].erase(p);
		sectors[x+1][y].erase(p);
		sectors[x+1][y+1].erase(p);
		sectors[x][y+1].erase(p);
	}
	
	function sectorHandler() 
	{
		mymc=_root.createEmptyMovieClip('linemc', 2);
		sectorSize=100;
		sectors=new Array();
		lines=new Array();
		onScreen=new Array();
	}
}