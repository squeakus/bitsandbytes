var Particle = function() {
    this.randomize = function(){
        this.initializing = true;
        this.xdir = Math.round((Math.random()*6)-3);
        this.ydir = Math.round((Math.random()*6)-3);
        this.xLoc = Math.round((Math.random()*canvas.width));
        this.yLoc = Math.round((Math.random()*canvas.height));
    }
    this.randomize();

    this.update = function(ctx, drawvert){
        if (this.xLoc > canvas.width ||
            this.yLoc > canvas.height ||
	    this.xLoc < 0 ||
            this.yLoc < 0){
	    this.randomize();
	}

	this.xLoc += this.xdir;
	this.yLoc += this.ydir;

	//draw the circle
	if (drawvert){
	    ctx.beginPath();
	    ctx.arc(this.xLoc, this.yLoc, 10, 0, 2*Math.PI, false);
	    ctx.fillStyle="green";
	    ctx.stroke();
	    ctx.fill();
	}
    }
    
    this.changedir = function(){
        this.xdir = Math.round((Math.random()*10)-5);
        this.ydir = Math.round((Math.random()*10)-5);
    }
}