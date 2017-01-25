function drawarrow(paper, x, y, speed, rot)
{
    len = speed * 3;
    y = y + (len/2);
    var xend = x;
    var yend = y - len;
    var direction = "r"+rot;
    arrowpath = "M "+x+" "+y+" L "+xend+" "+yend+" L "+(xend-2)+" "+(yend+2)+" M "+xend+" "+yend+" L "+(xend+2)+" "+(yend+2);
    var arrow = paper.path(arrowpath);
    arrow.attr({'stroke-width': 2,
		transform:"r90"
		});
}


window.onload = function() {  
    var paper = new Raphael(document.getElementById('canvas_container'), 500, 500); 
    drawarrow(paper, 250, 250, 5 , 90);
    drawarrow(paper, 300, 300, 0.3, 180);
    drawarrow(paper, 100, 100, 10 ,10);
    drawarrow(paper, 400, 400, 2 ,270);
}  
