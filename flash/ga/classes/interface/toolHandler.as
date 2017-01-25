class toolHandler
{
	var mymc:MovieClip;
	var p1, p2:cartesian;
	var tool:Number;
	var puindex:Number;
	var old:Number;
	var last:cartesian;
	var lineCol:Array=[0xFF0000,0x0000FF,0x000000,0xFF00FF,0x00FFFF,0xFFFF00, 0xFFAA00];
	var drawing:Boolean;
	var quick:Boolean;
	var t:Number;
	
	function press()
	{
		quick = (tool==0 || tool==4 || tool==6) && _root.key.quick;
		drawing=true;
		t=0;
		_root.cam.auto=false;
		if(_root.key.camera)
		{
			tool=2;
		}
		if(!quick && tool!=2 && tool!=1 && tool!=5){p1.equ(p2);}
		last=new cartesian(0,0);
	}
	
	function hold()
	{
		t++;
		if(tool!=2 && !quick)
		{
			var m=_root.mouse.pos.sub(_root.center);
			m.x=m.x*0.8;
			if(m.lenSqr()>30000 && t>20){_root.cam.pos.inc(m.factor(0.08))};
		}
	
		if(tool==0 || tool==4 || tool==6){p2.equ(_root.mouse.real);}
		
		if(tool==1){_root.sectors.erase(_root.mouse.real);}
		
		if(tool==3)
		{
			p2.inc(_root.mouse.real.sub(p2).factor(0.04));
			var ll=200+6*(_root.mouse.real.sub(p2).length());
			if(p2.sub(p1).lenSqr()>ll)
			{
				_root.sectors.newLine(p1.x, p1.y, p2.x, p2.y);
				last=p2.sub(p1);
				p1.equ(p2);
			}
		}
		
		if(tool==2){_root.cam.pos.inc(_root.mouse.old.pos.sub(_root.mouse.pos).factor(1/_root.cam.zoom))}
	}
	
	function release()
	{	
		p2=_root.mouse.real;
		drawing=false;
		
		if(tool==0){_root.sectors.newLine(p1.x, p1.y, p2.x, p2.y);}
		if(tool==4){_root.sectors.sceneryLine(p1.x, p1.y, p2.x, p2.y);}
		
		if(tool==5){_root.sectors.newTarget(p2.x, p2.y);}
		
		if(tool==6)
		{
			var angle=Math.round(360*Math.atan2(-(p2.x-p1.x),p2.y-p1.y)/(2*Math.PI));
			_root.sectors.newPowerup(p1.x, p1.y, puindex, angle);
		}
		
		if(tool!=3 && tool!=2 &&  tool!=1 && tool!=5 && tool!=6){p1.equ(p2);}
		
		tool=old;
	}
	
	function update()
	{
		quick = (tool==0 || tool==4) && _root.key.quick;
		if(!_root.gui.hover)
		{
			if(_root.mouse.press){press();}
		}
		
		if(drawing)
		{	
			if(_root.mouse.down){hold();}
			if(_root.mouse.release){release();}
		} else
		{
			p2=_root.mouse.real;
		}
	}
	
	function draw()
	{
		mymc.clear();
		
		if(_root.cam.auto){return}
		
		var s=p1.toScreen();
		mymc.lineStyle(0, 0x0000F0);
		mymc.moveTo(s.x-2, s.y-2);
		mymc.lineTo(s.x+2, s.y+2);
		mymc.moveTo(s.x-2, s.y+2);
		mymc.lineTo(s.x+2, s.y-2);
		
		if((tool==0  || tool==3 || tool==4 || tool==6) && (drawing || quick))
		{
			mymc.lineStyle(1, lineCol[tool]);
			var s1=p1.toScreen();
			var s2=p2.toScreen();
			mymc.moveTo(s1.x, s1.y);
			mymc.lineTo(s2.x, s2.y);
		}
		
		if(_root.mouse.grid)
		{
			s=_root.mouse.real.toScreen();
			mymc.lineStyle(0, 0xA000F0);
			
			mymc.moveTo(s.x-3, s.y-3);
			mymc.lineTo(s.x+3, s.y+3);
			mymc.moveTo(s.x-3, s.y+3);
			mymc.lineTo(s.x+3, s.y-3);
		}
		
	}
	
	function action(p:String)
	{
		if(_root.cam.auto){_root.cam.tzoom=0.6;}
		_root.cam.auto=false;
		if(p=='pencil'){tool=0;_root.gui.message('Tool: Line');}
		if(p=='rubber'){tool=1;_root.gui.message('Tool: Eraser');}
		if(p=='camera'){tool=2;_root.gui.message('Tip: Press and hold CTRL');_root.cam.auto=false;}
		if(p=='curve'){tool=3;_root.gui.message('Tool: Curve');}
		if(p=='scenery'){tool=4;_root.gui.message('Tool: Scenery');}
		if(p=='goal'){tool=5;_root.gui.message('Tool: Goal');}
		if(p=='powerup')
		{
			if(tool==6)
			{
				puindex++;
				if(puindex>4){puindex=1;}
			}
			tool=6;
			if(puindex==1){_root.gui.message('Tool: Slow Motion');}
			if(puindex==2){_root.gui.message('Tool: Gravity');}
			if(puindex==3){_root.gui.message('Tool: Boost');}
			if(puindex==4){_root.gui.message('Tool: Bomb');}
			_root.gui.changePowerup(puindex);
		}
		old=tool;
	}
	
	function toolHandler() 
	{
		mymc=_root.createEmptyMovieClip('toolmc', 15);
		tool=0;
		puindex=1;
		old=0;
		p1=new cartesian(40,50);
		p2=new cartesian(0,0);
		last=new cartesian(0,0);
	}
}
