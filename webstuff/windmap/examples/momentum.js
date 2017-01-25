var Vector = function(paper) {
    this.xloc = Math.floor((Math.random()*500)+1);
    this.yloc = Math.floor((Math.random()*500)+1);
    this.particle = paper.circle(this.xloc,this.yloc,1);
    this.particle.attr({'fill':'black'});
    this.xdir = 0;
    this.ydir = 5;

    this.update = function(){
	this.xloc += this.xdir;
	this.yloc += this.ydir;
    }

    this.randomize = function(){
	//this.xdir = Math.round((Math.random()*10)-5);
	//this.ydir = Math.round((Math.random()*10)-5);
    }
    this.randomize();
}

window.onload = function() {  

    var paper = new Raphael(document.getElementById('canvas_container'), 
			    500, 500); 
    var veclist = [];
    var counter = 0;
    //initialise points
    for (var i = 0; i < 100; i+= 1){
	veclist.push(new Vector(paper));    
    }

    function doAnimation() {
	counter += 1;
    	for (var i = 0; i < veclist.length; i++){
	    
    	    veclist[i].randomize();
    	    veclist[i].update();
	    veclist[i].particle.attr({cx: veclist[i].xloc, 
				      cy: veclist[i].yloc});

	    if (veclist[i].xloc > 500 || veclist[i].yloc > 500){
		veclist[i].particle.remove();
		veclist[i] = new Vector(paper);
	    }
	}
    }
    var timer = setInterval(function(){doAnimation()},10);
}   
