 window.onload = function() {
    var paper = new Raphael(document.getElementById('canvas_container'), 500, 500);
    var tetronimo = paper.path("M 250 250 l 0 -50 l -50 0 l 0 -50 l -50 0 l 0 50 l -50 0 l 0 50 z");
    tetronimo.attr(
        {
            gradient: '90-#526c7a-#64a0c1',
            stroke: '#3b4449',
            'stroke-width': 10,
            'stroke-linejoin': 'round',
	    transform:"r-90"
        }
    );
    // tetronimo.animate({transform:"r360"}, 2000, 'bounce');

    tetronimo.animate({transform:"r360", 'stroke-width': 1}, 2000, 'bounce', function() {
    /* callback after original animation finishes */
    this.animate({
        rotation: -90,
        stroke: '#3b4449',
        'stroke-width': 10
    }, 1000);
     });
     
     tetronimo.animate({  
     	 path: "M 250 250 l 0 -50 l -50 0 l 0 -50 l -100 0 l 0 50 l 50 0 l 0 50 z"  
     }, 5000, 'elastic');

     //paper.rect(100, 100, 300,300).animate({transform:"r-45,0,0"}, 2000);  
}