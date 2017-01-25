class sector
{
	var lines:Array;
	var i,j:Number;
	
	function addLine(pObj:Object) 
	{
		lines.push(pObj);
	}
	
	function resetDrawn() 
	{
		for( var k in lines ) { lines[k].drawn=false; }
	}
	
	function resetCollided() 
	{
		for( var k in lines ) { lines[k].collided=false; } 
	}
	
	function collide(pObj:Object)
	{
		for(var i in lines) { lines[i].collide(pObj); }
	}
	
	function erase(p:cartesian) 
	{
		for( var k in lines ) { lines[k].erase(p); }
	}
	
	function remove()
	{
		for(var i in lines){if( lines[i].remove ) {lines[i]=undefined};}
	}
	
	function draw(pMC:MovieClip) 
	{
		for(var i in lines) { lines[i].draw(pMC); }
	}
	
	function sector(pI:Number,pJ:Number) 
	{
		lines=new Array();
	}
}