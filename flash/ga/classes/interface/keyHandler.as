class keyHandler
{
	var swap:Boolean;
	var swap_lg:Boolean;
	var jump:Boolean;
	var jump_lg:Boolean;
	var accelerate:Boolean;
	var left:Boolean;
	var right:Boolean;
	var brake:Boolean;

	var w,s,a,d:Boolean;
	
	var camera:Boolean;
	var quick:Boolean;
	var grid:Boolean;
	var grid_lg:Boolean;
	
	function update()
	{
		if(Key.isDown(Key.ENTER)){console.action('restart');}
		
		accelerate=true;
		left=false;
		right=false;
		brake=false;
		camera=Key.isDown(Key.CONTROL);
		quick=Key.isDown(Key.SHIFT);
		
		if(Key.isDown(71))
		{
			if(grid_lg){_root.mouse.grid=!_root.mouse.grid}
			grid_lg=false;
			_root.gui.message(_root.mouse.grid ? "Grid on" : "Grid off");
		} else
		{
			grid_lg=true;
		}
		
		w=Key.isDown(87);
		s=Key.isDown(83);
		a=Key.isDown(65);
		d=Key.isDown(68);
	}
	
	function keyHandler() 
	{
	}
}