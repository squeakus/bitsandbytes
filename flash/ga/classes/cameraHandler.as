class cameraHandler
{
	var pos:cartesian;
	var zoom:Number=2;
	var auto:Boolean=true;
	var tzoom:Number=0.5;
	
	function update(){
		if(auto)
		{
			pos.x+=(_root.player.head.pos.x-pos.x)/5;
			pos.y+=(_root.player.head.pos.y-pos.y)/5;
		} 
		else
		{
			if (Key.isDown(187)){tzoom+=(2-tzoom)/20;}
			if (Key.isDown(189)){tzoom+=(0.2-tzoom)/5;}
			if(_root.key.a){pos.x+=-10/zoom}
			if(_root.key.d){pos.x+=10/zoom}
			if(_root.key.w){pos.y+=-10/zoom}
			if(_root.key.s){pos.y+=10/zoom}
		}
		
		zoom+=(tzoom-zoom)/5;
		
		if(isNaN(pos.x) || isNaN(pos.y)){pos=new cartesian(0,0);zoom=1;}
		if(isNaN(zoom)){zoom=1;}
		
	}
	
	function cameraHandler() {
		pos = new cartesian(0,0);
	}
}