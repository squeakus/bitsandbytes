var Particle = function(paper) {

    this.randomize = function(){
	this.initializing = true;
	this.xdir = Math.round((Math.random()*10)-5);
	this.ydir = Math.round((Math.random()*10)-5);
	this.x1 = Math.round((Math.random()*500));
	this.y1 = Math.round((Math.random()*650));
	this.x2 = this.x1;
	this.y2 = this.y1;
    }
    this.randomize();

    this.age = 0;
    this.nearest = '';
    this.xdir = 0;
    this.ydir = 0;
    this.particle = paper.circle(this.x1,this.y1,1);
    this.particle.attr({'fill':'black'});
    this.tail = paper.path(['M',this.x1, this.y1,
			    'l',this.x1-this.x2,this.y1-this.y2].join());

    //return magnitude of the vector
    this.mag = function() {
        return Math.sqrt(this.xdir * this.xdir + this.ydir * this.ydir);
    }

    //limit the max length of the vector
    this.setMag = function(length) {
        var current = this.mag();
        if (current > length) {
            var scale = length / current;
            this.xdir *= scale;
            this.ydir *= scale;
        }
    }

    //change particle location and redraw
    this.update = function(showtails){
	this.age += 1;
	if(showtails){
	    //draw the tails every 10 iters
	    if (this.age % 10 == 0){
		this.particle.attr({stroke:'none',
				    fill:'none'});
		this.tail.remove();
		this.tail = paper.path(['M',this.x1, this.y1,
					'l',this.x1-this.x2,this.y1-this.y2]
				       .join());
		this.x2 = this.x1;
		this.y2 = this.y1;
	    }
	}

	this.x1 += this.xdir;
	this.y1 += this.ydir;
	
	if (!showtails){
	    this.x2 = this.x1;
	    this.y2 = this.y1;

	    this.tail.attr({stroke:'none'});
	    this.particle.attr({stroke:'black',
				    fill:'black'});
	
	    this.particle.attr({cx: this.x1, 
				cy: this.y1});
	}
    }

    //find nearest weather station
    this.findNN = function(weather){
	var mindist = 10000;
	var mindist2 = 10000;
	var neighbour = weather[0];
	var neighbour2 = weather[1];

	//iterate the stations
	for(var i = 0; i < weather.length; i+=1) {
    	    var data = weather[i];
    	    var coords = data['mapcoord'];
	    var distance = Math.sqrt(Math.pow((coords[0]-this.x1),2) 
				     + Math.pow((coords[1]-this.y1),2));
	    if (distance < mindist){
		mindist2 = mindist;
		mindist = distance;
		neighbour2 = neighbour;
		neighbour = data;
	    }
	}
	// get station info
	var coords = neighbour['mapcoord'];
	var radangle = neighbour['windangle'] * (Math.PI/180);
	var scaledspeed = (neighbour['speed']/20)+1;
	var xwind = Math.sin(radangle) * scaledspeed;
	var ywind = -Math.cos(radangle) * scaledspeed;


	// start according to closest station
	if (this.initializing){
	    this.xdir = ywind;
	    this.ydir = xwind;
	    this.initializing = false;

	    // if (this.age == 50){
	    // 	console.log("speed: "+scaledspeed+" min: "+mindist);
	    // }
	}
	else{
	    if (mindist < 5){
		mindist = 5;
	    }
	    // incrementing xdir
	    if (this.xdir != xwind){
		if(this.xdir < xwind){
		    this.xdir += 1 / mindist;
		}
		else if(this.xdir > xwind){
		    this.xdir -= 1 / mindist;
		    
		}
	    }
	    // incrementing ydir
	    if (this.ydir != ywind){
		if(this.ydir < ywind){
		    this.ydir += 1 / mindist;
		}
		else if(this.ydir > ywind){
		    this.ydir -= 1 / mindist;
		}
	    }
	}
	// var steerx = (xvec - this.xdir) / distance;
	// var steery = (yvec - this.ydir) / distance;
	// this.xdir += steerx;
	// this.ydir += steery;
	// if (this.xdir > 5){ this.xdir = 5;}
	// if (this.ydir > 5){ this.ydir = 5;}
//	console.log(distance);
    }
}
