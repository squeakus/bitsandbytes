class parser
{
	static function point(p:cartesian)
	{
		return p.x.toString(32)+" "+p.y.toString(32);
		
	}

	static function getCode(p:sectorHandler)
	{
		var lines = p.lines;
	
		var output='';
		var count=lines.length
		
		if(count==undefined){return 'error: unknown'}
		
		for(var i in lines){lines[i].drawn=false}
		
		var i=0;
		while(i<count)
		{
			i=0;
			while (lines[i].drawn && i<count){i++}
			_root.halt=0;
			if(i<count){output+=point(lines[i].p1)+lines[i].getCode()+','}
		}
		
		output=output.slice(0, output.length-1);
		output+='#'
		output+='#'
				
		return output;
	}
	
	static function loadMap(p:String):sectorHandler
	{
		var legacy=false;
		
		var s=(p.split('#')[0]).split(' ');
		var o=(p.split('#')[0]).split(',');
		if(s.length<o.length && s.length>3)
		{
			legacy=true;
			
			var a='';
			for(var i=0; i<s.length; i++)
			{
				var s2=s[i].split(',')
				for(var j=0; j<s2.length; j++)
				{
					a+=s2[j]+' ';
				}
				a+=',';
			}
			
			a+='##';
			
			
			var t=(p.split('#')[1]).split(' ');
			for(var i=0; i<t.length; i++)
			{
				a+='T ';
				var t2=t[i].split(',')
				for(var j=0; j<t2.length; j++)
				{
					a+=t2[j]+' ';
				}
				a+=',';
			}
						
			p=a;
		}
		
		
		var output=new sectorHandler();
		
		s=p.split('#');
		var lines=s[0].split(',');
		
		for (var i=0; i<lines.length; i++)
		{
			s=lines[i].split(' ');
			if(s.length>=4)
			{
				var v=new Array();
				
				for (var j=0; j<s.length; j++){v.push(parseInt(s[j], 32))};
				
				var x=v[0];
				var y=v[1];
				var n=2;
				
				while(n<v.length-1)
				{
					output.newLine(x,y,v[n],v[n+1]);
					x=v[n];
					y=v[n+1];
					n+=2;
				}
			}
		}
		
		return output;
	}
}