class dna
{	
	var radius:Array;
	var px:Array;
	var py:Array;
	var dampConstant:Number;
	var springConstant:Number;
	var score:Number;
	
	function rnd(p)
	{
		return Math.random()*p;
	}
	
	static function crossVars(p1:Number, p2:Number)
	{
		var w=Math.random();
		return w*p1+(1-w)*p2;	
	}
	
	static function cross(m1:dna, m2:dna)
	{
		var w=Math.random();
		var output=new dna();
		
		for(var i=0; i<4; i++){output.radius[i]=crossVars(m1.radius[i],m2.radius[i]);}
		for(var i=0; i<4; i++){output.px[i]=crossVars(m1.px[i],m2.px[i]);}
		for(var i=0; i<4; i++){output.py[i]=crossVars(m1.py[i],m2.py[i]);}
		
		output.dampConstant=crossVars(m1.dampConstant, m2.dampConstant);	
		output.springConstant=crossVars(m1.springConstant, m2.springConstant);
		
		return output;
	}
	
	function mutate()
	{
		for(var i=0; i<4; i++){if(rnd(100)>95){radius[i]=5+rnd(20)}}
		for(var i=0; i<4; i++){if(rnd(100)>95){px[i]=rnd(150)}}
		for(var i=0; i<4; i++){if(rnd(100)>95){py[i]=rnd(100)}}
		
		if(rnd(100)>95){dampConstant=rnd(0.5)}
		if(rnd(100)>95){springConstant=rnd(0.5)}
	}
	
	function id()
	{
		var s='dna: r[';
		for(var i in radius){s+=Math.round(radius[i])+(i==0 ? ']' : ',')} 
		s+=' px[';
		for(var i in px){s+=Math.round(px[i])+(i==0 ? ']' : ',')} 
		s+=' py[';
		for(var i in py){s+=Math.round(py[i])+(i==0 ? ']' : ',')} 
		s+=' s['+Math.round(dampConstant*10)+','+Math.round(springConstant*10)+']';
		return s;
	}

	function dna()
	{
		radius=new Array();
		for(var i=0; i<4; i++){radius.push(5+rnd(20))}
		
		px=new Array();
		for(var i=0; i<4; i++){px.push(rnd(150))}
		
		py=new Array();
		for(var i=0; i<4; i++){py.push(rnd(100))}
		
		dampConstant=rnd(0.5);
		springConstant=rnd(0.5);
		
		score=undefined;
	}
}