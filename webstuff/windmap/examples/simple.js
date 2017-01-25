window.onload = function() {  
    var paper = Raphael(document.getElementById('canvas_container'), 500, 500); 
    var circle = paper.circle(100, 100, 80);  
    for(var i = 0; i < 5; i+=1) {  
	var multiplier = i*5;  
	paper.circle(250 + (2*multiplier), 100 + multiplier, 50 - multiplier);
    }  
    var rectangle = paper.rect(200, 200, 250, 100);    
    var ellipse = paper.ellipse(200, 400, 100, 50); 
    var wavyline = paper.path("M300,350l100,150");
//    wavyline.attr({fillStyle:"rgba(255, 255, 255, .05)",
//   		   'stroke-style':"rgba(255, 255, 255, .05)"});


    var t = paper.text(50, 10, "HTML5ROCKS");

    var letters = paper.print(150, 50, "HTML5ROCKS", paper.getFont("Vegur"), 40);
    letters[4].attr({fill:"orange"});
    // for (var i = 5; i < letters.length; i++) {
    // 	letters[i].attr({fill: "#3D5C9D", "stroke-width": "2", stroke: "#3D5C9D"});
    // }

    var tetronimo = paper.path("M 250 250 l 0 -50 l -50 0 l 0 -50 l -50 0 l 0 50 l -50 0 l 0 50 z");
    tetronimo.attr(  
        {  
            gradient: '90-#526c7a-#64a0c1',  
            stroke: '#3b4449',  
            'stroke-width': 10,  
            'stroke-linejoin': 'round',  
            rotation: -90  
        }  
    );
}  
