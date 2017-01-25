// what about right click
// get rid of mesharray at soonest convenience
// move check for filereader from glObject to meshrender
// glObject with mixed vars and this, terible idea?
// what about multiple listeners? should handlers all go in one place?

var spanIds = [];
var glObjects = {}; 
var blackWhite = true;
var maxObj;
var maximized = false;
// methods for handling user clicks

function minimizeWindow(link){
    maximized = false;
    rotateObjects();
    $(link).animate({height: '0px', width: '0px'}, 'easeInOutCubic', function(){link.parentNode.removeChild(link);delete maxObj;});
    
}

function selected(link) {
    var clickedColor = colorToHex(link.style.borderColor);
    if(clickedColor== "#6400"){
	link.style.width = 250;
	link.style.height = 250;
	dehighlight(link);
    }
    else{
	if(!maximized){
	    maximized = true;
	rotateObjects();
	link.style.borderColor = "#006400";
	clickedCanvas = link.id.substring(4);;
	clickedCanvas = "mesh"+ clickedCanvas;
	var clickedObj = glObjects[clickedCanvas];
	var verts = clickedObj.vertices;

	var max_frame = document.createElement('div');
	var pixels = window.innerHeight - 160;
	var margins = pixels/2;
	var pixelStr = pixels+'px';
	var marginStr = '-'+margins+'px';
	console.log("marginStr:"+marginStr);

	max_frame.id = "maximizeWindow";
	max_frame.style.marginTop = marginStr;
	max_frame.style.marginLeft = marginStr;
	max_frame.className = "maximizeFrame";
	max_frame.onclick = function(){minimizeWindow(this);};		
	max_frame.innerHTML=['<canvas id="maxCanvas" width="'+pixelStr+'" height="'+pixelStr+'" tabindex="1"></canvas>'].join('');
	document.body.appendChild(max_frame);
	$(max_frame).animate({height: pixelStr, width: pixelStr}, 'easeInOutCubic', function(){console.log("maximization complete")});

	//create the object
	maxObj = new glObject("maxCanvas", verts);
	maxObj.changeRotation();
	}
    }
}

function highlight(link) {
    link.style.color = "#A32B26";    
}

function dehighlight(link) {
    link.style.borderColor = "#000000";
}

function colorToHex(color) {
    if(color == "") return "#000000";
    if(color.substr(0, 1) === '#') return color;

    var digits = /(.*?)rgb\((\d+), (\d+), (\d+)\)/.exec(color);
    var red = parseInt(digits[2]);
    var green = parseInt(digits[3]);
    var blue = parseInt(digits[4]);
    
    var rgb = blue | (green << 8) | (red << 16);
    return digits[1] + '#' + rgb.toString(16);
};

function updateProgress(current, total) {

    var percentLoaded = 0;
    if(current == total - 1) percentLoaded = 100;
    else percentLoaded = (100 / total) * current;
    var percentLoaded = Math.round(percentLoaded);


    var progress = document.querySelector('.percent');
    if (percentLoaded < 100) {
	//console.log("\% loaded:"+percentLoaded);
        progress.style.width = percentLoaded + '%';
        progress.textContent = percentLoaded + '%';
    }
    else{
	//console.log("Finished");
	progress.style.width = '100%';
	progress.textContent = '100%';
    }
    setTimeout("document.getElementById('progress_bar').className='';", 2000);
}

function rotateObjects(){
    for(var key in glObjects){
	glObjects[key].changeRotation();
    }
}

function invertColors(){

    for(var key in glObjects){
     	glObjects[key].invertMeshColors(blackWhite);
    }
    if(blackWhite){
	blackWhite = false;
	backCol = '#ffffff';
    }
    else{
	blackWhite = true;
	backCol = '000000';
    }

    document.body.style.backgroundColor=backCol;

    var frameList = document.querySelectorAll('.canvasFrame');
    for(var i = 0; i < frameList.length; i++){
     	frameList[i].style.borderColor=backCol;
    }
}

function initMeshes(){
    glObjects = {};
    objCounter = 0;

    for (var i = 0, f; i < initpop.length; i++) {
	var vertices = null;
	vertices = initpop[i];
	
	var spanName = "span"+objCounter;
	var canvasName = "mesh"+objCounter;
	spanIds.push(spanName);
	
	//create span element
	var span = document.createElement('span');
	span.id = spanName;
	span.className = "canvasFrame";
	span.onclick = function(){selected(this);};		
	span.innerHTML=['<canvas id="',canvasName,'" width="250" height="250" tabindex="1"></canvas>'].join('');
	document.getElementById('main_frame').insertBefore(span, null);
	//create the object
	var obj = new glObject(canvasName, vertices);
	glObjects[canvasName] = obj
	objCounter++;
    }
}

// open file, read mesh, add to meshArray
function handleFileSelect(evt){
    evt.stopPropagation();
    evt.preventDefault();

    //creating files object
    var files = evt.dataTransfer.files;

    var progress = document.querySelector('.percent');
    progress.style.width = '0%';
    progress.textContent = '0%';

    if (!files.length) {
	alert('Please select a file!');
	return;
    }

    //delete old meshes
    while (spanIds.length > 0){
	var spanId = spanIds.pop();
	element = document.getElementById(spanId);
	element.parentNode.removeChild(element);
    }
    //clear mesh array and gl objects 
    glObjects = {};
    objCounter = 0;
    var filesRead = 0;

    for (var i = 0, f; i < files.length; i++) {
	f = files[i];
		
	// Only process meshes
	if (!f.name.match('\.glmesh'))continue;

	var reader = new FileReader();
	reader.onloadstart = function(e) {
	    document.getElementById('progress_bar').className = 'loading';
	};

	reader.onload =  (function(theFile) {
            return function(e) {
		//read in vertex values
		filesRead++;
		updateProgress(filesRead, files.length);

		var vertices = null;
		var mesh_string = e.target.result;
		mesh_string =  mesh_string.replace(/(\r\n|\n|\r)/gm,"");
		vertices = mesh_string.split(",");

		var spanName = "span"+objCounter;
		var canvasName = "mesh"+objCounter;
		spanIds.push(spanName);
		
		//create span element
		var span = document.createElement('span');
		span.id = spanName;
		span.className = "canvasFrame";
		span.onclick = function(){selected(this);};		
		span.innerHTML=['<canvas id="',canvasName,'" width="250" height="250" tabindex="1"></canvas>'].join('');
		document.getElementById('main_frame').insertBefore(span, null);
		//create the object
		var obj = new glObject(canvasName, vertices);
		glObjects[canvasName] = obj
		objCounter++;
	    };
	})(f);
	reader.readAsBinaryString(f);
    }
}

function handleDragOver(evt) {
    evt.stopPropagation();
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
  }


// set up listeners
function initListeners(){
    document.querySelector('.rotateButton').addEventListener('click', rotateObjects, false);
    document.querySelector('.invertButton').addEventListener('click', invertColors, false);

    // Setup the dnd listeners.
    var dropZone = document.getElementById('main_frame');
    dropZone.addEventListener('dragover', handleDragOver, false);
    dropZone.addEventListener('drop', handleFileSelect, false);
}

initMeshes();
initListeners();
