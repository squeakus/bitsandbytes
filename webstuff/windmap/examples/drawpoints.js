window.onload = function() {  
    var paper = new Raphael(document.getElementById('canvas_container'), 500, 650); 
    for(var i = 0; i <point_list.length; i+=1) {
	console.log(point_list[i])
	paper.circle(point_list[i][0],point_list[i][1], 2);
    }
}  
