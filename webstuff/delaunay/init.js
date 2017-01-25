// RequestAnimFrame: a browser API for getting smooth animations
window.requestAnimFrame = (function(){
  return  window.requestAnimationFrame       || 
		  window.webkitRequestAnimationFrame || 
		  window.mozRequestAnimationFrame    || 
		  window.oRequestAnimationFrame      || 
		  window.msRequestAnimationFrame     ||  
		  function( callback ){
			window.setTimeout(callback, 1000);
		  };
})();

window.onload = function() {
    var canvas = document.getElementById("canvas"),
    ctx = canvas.getContext("2d"),
    popsize = 100,
    particles = new Array(popsize),
    x, y

    //initialize
    for (var i=0;i < particles.length; i++) {
	particles[i] = new Particle();
    }

    function doAnimation(){
	//clearscreen
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	for(var i=0; i<particles.length;i++){
	    particles[i].update(ctx, false);
	    if (counter % 10 == 0)
		particles[i].changedir();
	}

	// generate and draw the triangles
	var triangles = triangulate(particles);
	for (var i=0;i < triangles.length; i++)
	    triangles[i].draw(ctx);
	counter += 1;
    }

    // requestanimframe for smoother animation
    function animloop() {
	doAnimation();
	requestAnimFrame(animloop);
    }
    
    // start her running
    var counter = 0;
    animloop();
}	
