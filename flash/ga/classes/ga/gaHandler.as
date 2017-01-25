class gaHandler
{	
	var mymc:MovieClip;
	var timer:Number;
	var gen:Array;
	var genSize:Number=20;
	var ind:Number;
	var generationCount:Number;
	
	var graph:Array;
	var means:Array;
	var best:Number
	
	function drawGraph()
	{
		var w=150;
		var h=50;
		var x=10;
		var y=340;
		mymc.clear();
		plot.rect(mymc, x, y, x+w, y+h, 0xFFFFFF, 0xAAAAAA, 50);
		
		var xs=w/graph.length;
		var ys=h/best;
		
		mymc.lineStyle(1,0x000000);
		mymc.moveTo(x,y+h);
		
		for(var i=0; i<graph.length; i++)
		{
			mymc.lineTo(x+(i+1)*xs, y+h-graph[i]*ys);
		}
		
		mymc.lineStyle(1,0x00AA00);
		mymc.moveTo(x,y+h);
		
		for(var i=0; i<graph.length; i++)
		{
			mymc.lineTo(x+(i+1)*xs, y+h-means[i]*ys);
		}
		mymc.gind.text='g:'+generationCount;
	}
	
	function select()
	{
		trace('select');
		
		gen.sortOn(["score"], Array.NUMERIC);
		gen.reverse();
		
		graph.push(gen[0].score);
		var m=0;
		for(var i in gen){m+=gen[i].score}
		means.push(m/gen.length);
		
		var output=new Array();
		for(var i=0; i<gen.length; i++)
		{
			var j = i % 5
			output.push(gen[j])
		}
		gen=output;
	}
	
	function randint(p)
	{
		return Math.floor(Math.random()*p);
	}
	
	function crossover()
	{
		trace('crossover');
		var output=new Array;
		var l=gen.length;
		
		while(output.length<l-1)
		{
			//choose pair and remove used items
			var i=randint(gen.length);
			var m1=gen[i];
			gen.splice(i,1);
			
			var j=randint(gen.length);
			var m2=gen[j];
			gen.splice(j,1);
			
			//push
			output.push(dna.cross(m1,m2));
			output.push(dna.cross(m1,m2));
		}
		
		gen=output;
	}
	
	function mutate()
	{
		trace('mutate');
		for(var i in gen){gen[i].mutate()}
	}
	
	function nextgen()
	{
		trace('-----------------');
		generationCount+=1;
		trace('GENERATION #'+generationCount);
		select();
		crossover();
		mutate();
		
		drawGraph();
		
		ind=0;
	}

	function score(t)
	{
		trace(', '+t + ' after ' + Math.floor(_root.player.best) + ' units', true);
		gen[ind].score=Math.floor(_root.player.best);
		best = gen[ind].score>best ? gen[ind].score : best;
		ind++;
		if(ind>genSize-1){nextgen()}
		next();
	}
	
	function next()
	{
		trace('car '+(ind+1)+'/'+gen.length);
		game.restart();
		_root.player=new vehicle(gen[ind]);
	}
	
	function update()
	{
		if(timer++>1000)
		{
			score('timed out');
			timer=0;
		}
	}
	
	function gaHandler()
	{
		timer=0;
		
		mymc=_root.createEmptyMovieClip("gamc", 4);
		
		//create first gen
		trace('building primordial soup...done');
		
		gen=new Array();
		means=new Array();
		graph=new Array();
		best=0;
		generationCount=1;
		for(var i=0; i<genSize; i++){gen.push(new dna())}
		trace('------------------');
		ind=0;
		trace('GENERATION #1');
		
		
		mymc.createTextField('gind',0,10,338,100,350);
		var format=new TextFormat()
		format.font='_typewriter';
		format.size=10;
		format.align='left';
		format.color=0x999999;
		mymc.gind.setNewTextFormat(format);
		mymc.gind.border=false;
		mymc.gind.selectable=false;
		mymc.gind.text='g:1';

		next();
		drawGraph();
		
	}
}