class console
{	
	//write: the generic 'trace' function
	static function write(pText, newln) 
	{
		var a = newln==true ? '' : '\n';
		_root.consoletext.text += a+pText;
		//overfill
		var s=_root.consoletext.text.split('\r')
		if (s.length>23)
		{
			_root.consoletext.text=_root.consoletext.text.substr(s[0].length+1);
		}
	}
	
	//action: responds to an 'action string' passed from a button
	static function action(p:String)
	{
		if(p==undefined){return;}
		//trace('> '+p);
		//look for tool commands
		if(p.slice(0,2)=='t:'){_root.tools.action(p.slice(2));}
		//global commands
		if(p=='pause'){_root.gamePause=!_root.gamePause}
		if(p=='restartga')
		{
			game.restartga();
			_root.gui.msgPanel.time=0;
		}
		if(p=='grid')
		{
			_root.mouse.grid=!_root.mouse.grid
			var a=_root.mouse.grid ? 'on' : 'off'
			_root.gui.message('Grid is '+a);
		}
		
		if(p=='focus')
		{
			_root.cam.auto=!_root.cam.auto
			var a=_root.cam.auto ? 'Car' : 'Map'
			_root.gui.message('Focus: '+a);
		}
		
		if(p=='console')
		{
			_root.consoletext._visible=!_root.consoletext._visible;
		}
		
		if(p=='kill')
		{
			_root.ga.score('killed by player');
			_root.gui.message('You killed a car!', 0xFFAAAA);
		}
		
		if(p=='d:clear'){_root.gui.clearDialog()}
	}
		
	//startup: creates my textbox etc.
	static function startup() 
	{
		_root.lineStyle(1, 0xEEEEEE);
		_root.moveTo(27, 0);

		_root.createTextField('consoletext',0,10,0,470,400);
		
		var format=new TextFormat()
		format.font='_typewriter';
		format.size=11;
		format.align='left';
		format.color=0x99bb99;
		_root.consoletext.setNewTextFormat(format);
		_root.consoletext.border=false;
		_root.consoletext.selectable=false;
	}
}