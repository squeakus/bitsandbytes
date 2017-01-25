class guiHandler
{
	var mymc:MovieClip;
	var rightPanel:panel;
	var msgPanel:msg;
	var hover:Boolean;
	
	function message(p:String, pCol)
	{
		msgPanel.message(p, pCol);
	}
	
	function standardFormat()
	{
		var format = new TextFormat();
		format.font = '_sans';
		format.size = 14;
		format.align = 'left';
		format.color = 0xFFFFFF;
		return format;
	}
	
	function formatLabel(p:TextField, b:Boolean)
	{
		p.selectable=false;
		p.wordWrap=true;
		var format = standardFormat();
		if(b){format.bold=true};
		p.setNewTextFormat(format);		
	}
	
	function formatTextBox(p:TextField)
	{
		p.selectable = true;
		p.wordWrap=true;
		p.border=true;
		p.borderColor=0x888888;
		p.background=true;
		p.backgroundColor=0x777777;
		var format=standardFormat();
		format.size = 10;
		format.font = '_typewriter';
		p.setNewTextFormat(format);
	}
	
	function clearDialog()
	{	
		_root.dialog=true;
		mymc.splash.removeMovieClip();
		mymc.splash=undefined;
		
		var dmc = mymc.createEmptyMovieClip('dialog', mymc.getNextHighestDepth());
		plot.rect(dmc, 95, 145, 405, 255, 0xDDDDDD, 0xEEEEEE);
		plot.rect(dmc, 100, 150, 400, 250, 0x000000, 0x888888);
		
		dmc.createTextField("title", 1, 110, 160, 280, 20);
		dmc.createTextField("label", 2, 110, 180, 280, 20);
		formatLabel(dmc.title, true);
		formatLabel(dmc.label);
		dmc.title.text = 'Clear Map';
		dmc.label.text = 'Are you sure you want to clear this map?';
		
		dmc.createEmptyMovieClip('yes', 3);
		dmc.createEmptyMovieClip('no', 4);
		plot.button(dmc.yes, 'Yes');
		plot.button(dmc.no, 'No');
		dmc.yes._x=100; dmc.no._x=250; dmc.yes._y=dmc.no._y=220;
			
		dmc.yes.onPress=function()
		{
			_root.gui.mymc.dialog.removeMovieClip();
			_root.dialog=false;
			_root.targetCount=0;
			_root.sectors.kill();
			_root.sectors=new sectorHandler();
			_root.sectors.newLine(-40,50,200,50);
			game.restart();
			_root.gui.msgPanel.time=0;
			_root.tools.tool=_root.tools.old=0;
			_root.tools.p1=new cartesian(40,50);
		}
		
		dmc.no.onPress=function()
		{
			_root.gui.mymc.dialog.removeMovieClip();
			_root.dialog=false;
		}
	}
	
	function update()
	{	
		rightPanel.visible=_root.mouse.pos.x>470 && !_root.tools.drawing && !_root.tools.quick;
		
		rightPanel.update();
		msgPanel.update();
		
		hover = rightPanel.visible
		
		mymc.pausemc._visible=_root.gamePause;
	}
	
	function guiHandler() 
	{	
		mymc=_root.createEmptyMovieClip("guimc", 5);
		
		var temp=new Array(15);
		temp[0]=new menuItem('pencil', 'Draw lines', 't:pencil');
		temp[1]=new menuItem('rubber', 'Eraser', 't:rubber');
		temp[2]=new menuItem('grid', 'Grid', 'grid');
		temp[3]=new menuItem('pause', 'Pause', 'pause');		
		temp[11]=new menuItem('clear', 'Clear Map', 'd:clear');
		temp[5]=new menuItem('magnify', 'Move camera', 't:camera');
		
		temp[7]=new menuItem('console', 'Console', 'console');
		
		temp[6]=new menuItem('camera', 'Change Focus', 'focus');
		
		temp[8]=new menuItem('skull', 'Kill', 'kill');
		
		temp[9]=new menuItem('restart', 'Restart GA', 'restartga');

		rightPanel=new panel(mymc, 'rightPanel', 'right', temp);
		msgPanel=new msg(mymc, 'msgPanel');
		
		mymc.createEmptyMovieClip('pausemc',  mymc.getNextHighestDepth());
		plot.rect(mymc.pausemc, 242, 190, 247, 210, 0xCCFF88, 0x000000, 100)
		plot.rect(mymc.pausemc, 253, 190, 258, 210, 0xCCFF88, 0x000000, 100)
	}
}