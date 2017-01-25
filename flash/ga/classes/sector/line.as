class line
{
	var p1, p2:cartesian;
	var pp:cartesian;
	var length:Number;
	var drawn:Boolean;
	var collided:Boolean;
	var remove:Boolean;
	
	function draw(pMC:MovieClip) 
	{
		if(drawn){return}
		drawn=true;
		
		var d1=p1.toScreen();
		var d2=p2.toScreen();
		
		pMC.lineStyle(2*_root.cam.zoom, 0x000000, 100);
		pMC.moveTo(d1.x, d1.y);
		pMC.lineTo(d2.x, d2.y);
	}
	
	function collide(pMass:Object)
	{
		if(collided){return}
		collided=true;
		
		var m=pMass.pos;
		var v=pMass.vel;
		var r=pMass.radius;
		var d=new cartesian(0,0);
		var dl=0;
		var mp1=m.sub(p1);
		
		var k=mp1.dot(pp)/length/length;
		
		if(k>=0&&k<=1)
		{
			var cross=(mp1.x*pp.y-mp1.y*pp.x)*((mp1.x-v.x)*pp.y-(mp1.y-v.y)*pp.x)<0 ? -1 : 1;
			d = mp1.sub(pp.factor(k));
			dl = d.length()	//***note: this looks like it can be optimized
			if(dl<r||cross<0)
			{
				m.inc(d.factor((r*cross-dl)/dl));	//constrain
				pMass.drive(new cartesian(-d.y/dl, d.x/dl));	//drive
				return;
			}
		}
		
		if(k*length<-r || k*length>length+r){return}
		
		var end=k>0 ? p2 : p1
		d = m.sub(end);
		dl = d.length()	//***note: this looks like it can be optimized
		if(dl<r)
		{
			m.inc(d.factor((r-dl)/dl));	//constrain
			pMass.drive(new cartesian(-d.y/dl, d.x/dl));	//drive
			return;
		}
	}
	
	function erase(m:cartesian)
	{
		var r=15;
		var d=new cartesian(0,0);
		var dl=0;
		var mp1=m.sub(p1);
		
		var k=mp1.dot(pp)/length/length;
		
		if(k>=0&&k<=1)
		{
			d = mp1.sub(pp.factor(k));
			dl = d.length()
			if(dl<r)
			{
				remove=true;
				_root.sectors.remove(p1,p2);
			}
		}
	}
	
	function line(pX1:Number,pY1:Number,pX2:Number,pY2:Number) 
	{
		p1=new cartesian(Math.round(pX1),Math.round(pY1));
		p2=new cartesian(Math.round(pX2),Math.round(pY2));
		
		pp=p2.sub(p1);
		length=pp.length();
		remove=length<3||length>100000;
	}
}