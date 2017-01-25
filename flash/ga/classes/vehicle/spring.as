class spring
{	
	var parent:vehicle;
	var m1:Object;
	var m2:Object;
	var lrest:Number;
	var leff:Number
	var dampConstant:Number;
	var springConstant:Number;
	
	function draw()
	{
		var p1=m1.pos.toScreen();
		var p2=m2.pos.toScreen();
		
		parent.mymc.lineStyle(1,0xAAAAAA);
		parent.mymc.moveTo(p1.x,p1.y);
		parent.mymc.lineTo(p2.x,p2.y);
	}
	
	function contract(p1:Number, p2:Number)
	{
		leff+=(lrest-p1-leff)/p2;
	}
	
	function rotate(p:Number)
	{
		var mm = (m2.pos).sub(m1.pos);
		var n=new cartesian(-mm.y/leff, mm.x/leff);
		m1.pos.inc(n.factor(p));
		m2.pos.inc(n.factor(-p));
	}
	
	function update()
	{	
		var mm = (m2.pos).sub(m1.pos);
		var sep = mm.length();
		if(sep<1){return}	//****Note: this can be dealt with better
		mm=mm.factor(1/sep);
		var tens = mm.factor((sep-leff)*springConstant);
		var damp = m2.vel.sub(m1.vel).dot(mm)*dampConstant;
		tens.inc(mm.factor(damp));
		m2.vel.inc(tens.factor(-1));
		m1.vel.inc(tens);
	}
	
	function spring(pm1:Object, pm2:Object, pParent:vehicle) 
	{
		m1=pm1;
		m2=pm2;
		parent=pParent;
		leff = lrest = m2.pos.sub(m1.pos).length();;
		dampConstant = 0.5;
		springConstant = 0.7;
	}
}