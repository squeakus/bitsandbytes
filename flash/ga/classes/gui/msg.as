class msg
{
	var mymc1:MovieClip;
	var visible:Boolean;
	var time:Number;
		
	function message(p:String, pCol)
	{
		if(mymc1.label.text!=p || !visible)
		{
			visible=true;
			mymc1.clear();
			if(pCol==undefined){pCol=0xFFFFBB;}
			plot.rect(mymc1, -100,0,100,20, pCol, pCol);
			mymc1.label.text=p;
			time=70;
		}
	}
	
	function animate()
	{	
		mymc1._visible=visible;
		
		if(time>0){--time;}
		visible=time>0;
	}
	
	function update()
	{
		animate();
	}
	
	function msg(pMC, pName)
	{	
		visible=false;
		time=0;

		mymc1=pMC.createEmptyMovieClip(pName, pMC.getNextHighestDepth());
		mymc1._x=250;
		mymc1._y=1;
				
		plot.rect(mymc1, -100,0,100,20, 0xFFFFDD, 0xFFFFDD);
		plot.labelMsg(mymc1, -100);
	}
}