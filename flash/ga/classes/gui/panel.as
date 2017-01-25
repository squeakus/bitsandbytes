class panel
{
	var structure:Array;
	var mymc:MovieClip;
	var iconmc:MovieClip;
	var arrowmc:MovieClip;
	var tipmc:MovieClip;
	var visible:Boolean;
	var align:String;
	var arrowx1, arrowx2:Number;
	var hover:Number;
	var tipWidth:Number=110;
	
	function animate()
	{
		iconmc._visible=visible;
		
		var t=Math.floor(_root.mouse.pos.y/25);
		tipmc._y=t*25;
		tipmc._visible=visible && hover>30 && structure[t]!=undefined;
		tipmc.label.text=structure[t].tip;
		
		arrowmc._x = visible ? arrowx1 : arrowx2;
		
		arrowmc._alpha += visible ? (-arrowmc._alpha)/4 : (100-arrowmc._alpha)/15;
		if (_root.gui.dialog){arrowmc._alpha=0}
	}
	
	function update()
	{
		if(visible){hover = hover<100 ? hover+1 : hover} else {hover=0}
		animate();
		var t=Math.floor(_root.mouse.pos.y/25);
		if(_root.mouse.press && visible){console.action(structure[t].action);}
	}
	
	function drawIcons()
	{
		var depth=1;
		for (var i=0; i<=15; i++)
		{
			if(structure[i]!=undefined)
			{
				var mc=iconmc.attachMovie(structure[i].image, 'i_'+depth, depth);
				mc._x=-2;
				mc._y=(depth-1)*25-2;
				mc.stop();
			}
			depth++;
		}
	}
	
	function panel(pMC, pName, pAlign, pStructure)
	{	
		visible=false;
		align=pAlign;
		structure=pStructure;
		arrowx1 = align=='left' ? 24 : -6;
		arrowx2 = align=='left' ? 1 : 18;
		hover=0;
		
		mymc=pMC.createEmptyMovieClip(pName, pMC.getNextHighestDepth());
		
		mymc._x = align=='left' ? 0 : _root._width-27;
		
		tipmc=mymc.createEmptyMovieClip('tipmc', mymc.getNextHighestDepth());
		if(align=='right'){plot.tipRight(tipmc,0,0,tipWidth,25);} else {plot.tipLeft(tipmc,0,0,tipWidth,25);}
		tipmc._x = align=='left' ? 25 : -tipWidth;
		plot.labelTip(tipmc, 0, 2, tipWidth);
		
		iconmc=mymc.createEmptyMovieClip('iconmc', mymc.getNextHighestDepth());
		iconmc._visible=false;
		plot.rect(iconmc,0,0,24,399);
		
		drawIcons();
		
		var image = align=='right' ? 'smallArrow_l' : 'smallArrow_r';
		arrowmc=mymc.attachMovie(image, 'arrow', mymc.getNextHighestDepth());
		arrowmc._y=200-4;
	}
}