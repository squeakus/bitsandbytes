$(function(){
	var paper = Raphael('map', 500, 650);
	var popsize = 200;
	var tails = false;
	attributes = {
		fill: '#fff',
		stroke: '#3899E6',
		'stroke-width': 1,
		'stroke-linejoin': 'round'
	} 
	var weatherdata; 
	readWeather(false);
	var counties = new Array();
	var veclist = [];
	var t1 = paper.text(100, 10, "Press 'enter' to show weather stations");
	var t2 = paper.text(100, 20, "Press 't' to show tails");

    // handle keypresses
    document.onkeydown = handleKeyDown;
//    document.onkeyup = handleKeyUp;
var currentlyPressedKeys = {};

function handleKeyDown(event) {
	if (currentlyPressedKeys[event.keyCode] == true){
		currentlyPressedKeys[event.keyCode] = false;
	}
	else{
		currentlyPressedKeys[event.keyCode] = true;
	}
}

function handleKeyUp(event) {
	currentlyPressedKeys[event.keyCode] = false;
}
function readWeather(asyncVal) {
	console.log("called");
	$.ajax({
		type: 'GET',
		async: asyncVal,
		dataType: 'JSON',
		url: "data/weather.json", 
		success: function(data, textStatus, jqXHR){
			weatherdata = $.parseJSON(data);
		}
	});
}
function handleKeys() {
	// enter: draw wind directions
	if (currentlyPressedKeys[13]) {
		drawarrows(paper, weatherdata);
	}
	else if (!currentlyPressedKeys[13]) {
		deletearrows();
	}
	// t for tails
	if (currentlyPressedKeys[84]) {
		tails = true;
	}
	else if (!currentlyPressedKeys[84]) {
		tails = false;
	}

}

    // draw out the paths for each county
    for (var county in paths) {
    	var obj = paper.path(paths[county].path);
    	obj.attr(attributes);
    	counties[obj.id] = county;
    	obj
    	.hover(function(){
    		this.animate({
    			fill: '#1669AD'
    		}, 300);
    	}, function(){
    		this.animate({
    			fill: attributes.fill
    		}, 300);
    	})
    	.click(function(){
    		document.location.hash = counties[this.id];

    		var point = this.getBBox(0);
    		$('#map').next('.point').remove();		
    		$('#map').after($('<div />').addClass('point'));
    		$('.point')
    		.html(paths[counties[this.id]].name)
    		.prepend($('<a />').attr('href', '#').addClass('close').text('Close'))
    		.css({
    			left: point.x+(point.width/2)-80,
    			top: point.y+(point.height/2)-20
    		})
    		.fadeIn();		
    	});
    	$('.point').find('.close').live('click', function(){
    		var t = $(this),
    		parent = t.parent('.point');

    		parent.fadeOut(function(){
    			parent.remove();
    		});
    		return false;
    	});
    }
    //calculate the neighbours
    triangles = triangulate(weatherdata);
    console.log("tricount"+triangles.length);
    
    var neighbourhood = {};
    for(var i = 0; i < triangles.length; i++){
    	var a = triangles[i].a;
    	var b = String(triangles[i].b);
    	var c = String(triangles[i].c);
    	if (neighbourhood[a] == null){
    		console.log(a.x);
    		neighbourhood[a] = [];
    	}
    }
    console.log(neighbourhood);
    //draw the particles
    function doAnimation() {
    	handleKeys();
	//add a new particle on every frame
	if (veclist.length < popsize){
		veclist.push(new Particle(paper));    
	}
	for (var i = 0; i < veclist.length; i++){
		veclist[i].findNN(weatherdata);
		veclist[i].update(tails);
		if (veclist[i].age > popsize){
			veclist[i].randomize();
			veclist[i].age = 0;
		}
	}
}
setInterval(function(){readWeather(true)}, 1000000);
var timer = setInterval(function(){doAnimation()},10);
});

