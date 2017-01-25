class plot
{
	
	static var kx:Array=new Array(1.000, 0.924, 0.707, 0.383, 0.000, -0.383, -0.707, -0.924, -1.000, -0.924, -0.707, -0.383, -0.000, 0.383, 0.707, 0.924, 1.000);
	static var ky:Array=new Array(0.000, 0.383, 0.707, 0.924, 1.000, 0.924, 0.707, 0.383, 0.000, -0.383, -0.707, -0.924, -1.000, -0.924, -0.707, -0.383, 0.000);

	static function rect(pMC, x1, y1, x2, y2, col, col2, alpha)
	{
		if (col==undefined){col=0x000000;}
		if (col2==undefined){col2=0x000000;}
		if (alpha==undefined){alpha=100;}
		pMC.lineStyle(1, col2, alpha);
		pMC.beginFill(col, alpha);
		pMC.moveTo(x1, y1);
		pMC.lineTo(x2, y1);
		pMC.lineTo(x2, y2);
		pMC.lineTo(x1, y2);
		pMC.lineTo(x1, y1);
		pMC.endFill();
	}
	
	static function circle(pMC, cx, cy, r, col)
	{
		pMC.lineStyle(1, col);
		pMC.moveTo(cx+r, cy);
		var a=0;
		while(a++<16){pMC.lineTo(cx+r*kx[a], cy+r*ky[a])}
	}
	
	static function dircle(pMC, cx, cy, r, col)
	{
		pMC.lineStyle(1, col);
		pMC.moveTo(cx+r, cy);
		var a=0;
		while(a++<16)
		{
			a/2==Math.floor(a/2) ? pMC.lineTo(cx+r*kx[a], cy+r*ky[a]) : pMC.moveTo(cx+r*kx[a], cy+r*ky[a]);
		}
	}
	
	static function wheel(pMC, cx, cy, r, col, angle)
	{
		pMC.lineStyle(1, col);
		pMC.moveTo(cx+r, cy);
		var a=0;
		while(a++<16){pMC.lineTo(cx+r*kx[a], cy+r*ky[a])}
		
		pMC.lineStyle(1, 0xAAAAAA);
		pMC.moveTo(cx-r*Math.cos(angle), cy-r*Math.sin(angle));
		pMC.lineTo(cx+r*Math.cos(angle), cy+r*Math.sin(angle));
	}
	
		
	static function tipLeft(pMC)
	{
		pMC.lineStyle(1, 0xFFFFAA);
		pMC.beginFill(0xFFFFAA);
		pMC.moveTo(12, 1);
		pMC.lineTo(110, 1);
		pMC.lineTo(110, 23);
		pMC.lineTo(12, 23);
		pMC.lineTo(0, 12);
		pMC.lineTo(12, 1);
		pMC.endFill();
	}
	
	static function tipRight(pMC)
	{
		pMC.lineStyle(1, 0xFFFFAA);
		pMC.beginFill(0xFFFFAA);
		pMC.moveTo(0, 1);
		pMC.lineTo(98, 1);
		pMC.lineTo(110, 12);
		pMC.lineTo(98, 23);
		pMC.lineTo(0, 23);
		pMC.lineTo(0, 1);
		pMC.endFill();
	}
	
	static function standardFormat():TextFormat
	{
		var o=new TextFormat();
		o.font = '_sans';
		o.size = 14;
		o.align = 'center';
		o.color = 0x000000;
		return o;
		
	}
	
	static function labelMsg(pMC, x, y):TextField
	{
		pMC.createTextField("label", 1, x, y, 200, 20);
		var field = pMC.label;
		field.selectable = false;
		field.setNewTextFormat(standardFormat());
		field.text = '';
		return field;
	}
	
	static function labelTip(pMC, x, y, pWidth):TextField
	{
		pMC.createTextField("label", 2, x, y, pWidth, 20);
		var field = pMC.label;
		field.selectable = false;
		field.setNewTextFormat(standardFormat());
		field.text = '';
		return field;
	}
	
	static function button(pMC, p)
	{
		rect(pMC, 0, 0, 150, 30, 0x000000, 0x888888);
		pMC.createTextField("label", 2, 0, 5, 150, 25);
		var field = pMC.label;
		field.selectable = false;
		var format = standardFormat();
		format.color = 0xFFFFFF;
		format.bold = true;
		field.setNewTextFormat(format);
		field.text = p;
	}
	
	static function closeButton(pMC, p)
	{
		rect(pMC, 0, 0, 20, 20, 0x000000, 0x000000);
		pMC.lineStyle(5, 0xFFFFFF);
		pMC.moveTo(0,0);
		pMC.lineTo(20,20);
		pMC.moveTo(20,0);
		pMC.lineTo(0,20);
		
	}
}