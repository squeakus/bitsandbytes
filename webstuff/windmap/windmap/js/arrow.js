var arrowlist = [];
var drawn = false;

function drawarrows(paper, weatherdata){
    if (drawn == false){
	for(var i = 0; i < weatherdata.length; i+=1) {
    	    var data = weatherdata[i];
    	    var coords = data['mapcoord'];
            drawarrow(paper, coords[0],coords[1],
		      data['speed'], 
		      data['windangle']);
	}
	drawn = true;
    }
}

function drawarrow(paper, x, y, speed, rot){
    var speed = parseFloat(speed)+3;
    y = y + (speed/2);
    var xend = x;
    var yend = y - speed;
    var direction = "r"+rot;
    
    arrowpath = "M "+x+" "+y+" L "+xend+" "+yend+" L "+(xend-2)+" "+(yend+2)+" M "+xend+" "+yend+" L "+(xend+2)+" "+(yend+2);
    arrowlist.push(paper.path(arrowpath).attr({'stroke-width': 2,
					    transform:direction
					   }));
}

function deletearrows(){
    if(drawn == true){
	for(var i=0; i < arrowlist.length; i++){
	    arrowlist[i].remove();
	}
	drawn = false;
    }
}
