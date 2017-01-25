function glObject(cId, verts)
{
    this.canvasId = cId;
    this.vertices = verts;
    var gl;
    var gl_context;
    var shaderProgram;
    var mvMatrixStack = [];
    var lineVertexPositionBuffer;
    var lineVertexColorBuffer;
    var lastTime = 0;
    var currentlyPressedKeys = {};   
    var xRot = 0;
    var xSpeed = 0;
    var yRot = 0;
    var ySpeed = 100;
    var zSpeed = -2.0;
    var zDist = -2;
    var mouseDown = false;
    var lastMouseX = null;
    var lastMouseY = null;
    var rotateBool = true;
    var lineColor = [1.0, 1.0, 1.0, 1.0];
    var mvMatrix = mat4.create();
    var pMatrix = mat4.create();
    var mouseRotationMatrix = mat4.create();
    mat4.identity(mouseRotationMatrix);

    // public method for enabling rotation
    this.changeRotation = function() {
	rotateBool = !rotateBool ;	
    }

    //change default colors on the shaders
    this.invertMeshColors = function(blackWhite) {
	if(blackWhite){
	    gl_context.clearColor(1.0, 1.0, 1.0, 1.0);
	    lineColor = [0.0, 0.0, 0.0, 1.0];
	}
	else{
	    gl_context.clearColor(0.0, 0.0, 0.0, 1.0);
	    lineColor = [1.0, 1.0, 1.0, 1.0];
	}
	var colors = [];
	for (var i=0; i < vert_count; i++) {
            colors = colors.concat(lineColor);
        }
	this.initBuffers();
    }

    function initGL(canvas) {
	try {
            gl = canvas.getContext("webgl");
            gl.viewportWidth = canvas.width;
            gl.viewportHeight = canvas.height;

	} catch (e) {
        }
        if (!gl) {
            alert("Could not initialise WebGL, sorry :-(");
        }
	return gl;
    }
    
    function getShader(gl, id) {
	var shaderScript = document.getElementById(id);
	if (!shaderScript) {
            return null;
	}
	
	var str = "";
	var k = shaderScript.firstChild;
	while (k) {
            if (k.nodeType == 3) {
		str += k.textContent;
            }
            k = k.nextSibling;
	}
	
	var shader;
	
	if (shaderScript.type == "x-shader/x-fragment") {
            shader = gl.createShader(gl.FRAGMENT_SHADER);
	} else if (shaderScript.type == "x-shader/x-vertex") {
            shader = gl.createShader(gl.VERTEX_SHADER);
	} else {
            return null;
	}
	
	gl.shaderSource(shader, str);
	gl.compileShader(shader);
	
	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(shader));
            return null;
	}
	
	return shader;
    }
    
    function initShaders() {
	var fragmentShader = getShader(gl, "shader-fs");
	var vertexShader = getShader(gl, "shader-vs");
	
	shaderProgram = gl.createProgram();
	gl.attachShader(shaderProgram, vertexShader);
	gl.attachShader(shaderProgram, fragmentShader);
	gl.linkProgram(shaderProgram);
	if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
            alert("Could not initialise shaders");
	}
	gl.useProgram(shaderProgram);

	//coloring those vertices
 	shaderProgram.vertexColorAttribute = gl.getAttribLocation(shaderProgram, "aVertexColor");
	gl.enableVertexAttribArray(shaderProgram.vertexColorAttribute);

	shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
	gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);

	shaderProgram.pMatrixUniform = gl.getUniformLocation(shaderProgram, "uPMatrix");
	shaderProgram.mvMatrixUniform = gl.getUniformLocation(shaderProgram, "uMVMatrix");
	return gl;
    }
    
    function mvPushMatrix() {
	var copy = mat4.create();
	mat4.set(mvMatrix, copy);
	mvMatrixStack.push(copy);
    }
    
    function mvPopMatrix() {
	if (mvMatrixStack.length == 0) {
	    throw "Invalid popMatrix!";
	}
	mvMatrix = mvMatrixStack.pop();
    }
    
    function setMatrixUniforms() {
	gl.uniformMatrix4fv(shaderProgram.pMatrixUniform, false, pMatrix);
	gl.uniformMatrix4fv(shaderProgram.mvMatrixUniform, false, mvMatrix);
    }
    
    function degToRad(degrees) {
	return degrees * Math.PI / 180;
    }
    
    function handleMouseDown(event) {
        mouseDown = true;
        lastMouseX = event.clientX;
        lastMouseY = event.clientY;
    }

    function handleMouseUp(event) {
        mouseDown = false;
    }

    function handleMouseMove(event) {
        if (!mouseDown) {
            return;
        }

        var newX = event.clientX;
        var newY = event.clientY;
	
        var deltaX = newX - lastMouseX
        var newRotationMatrix = mat4.create();
        mat4.identity(newRotationMatrix);
        mat4.rotate(newRotationMatrix, degToRad(deltaX * 2), [0, 1, 0]);
        var deltaY = newY - lastMouseY;
        mat4.rotate(newRotationMatrix, degToRad(deltaY * 2), [1, 0, 0]);

        mat4.multiply(newRotationMatrix, mouseRotationMatrix, mouseRotationMatrix);
        lastMouseX = newX
        lastMouseY = newY;
    }

    this.initBuffers = function(){

	if (window.File && window.FileReader && window.FileList && window.Blob){
	} else {
	    alert('The File APIs are not fully supported in this browser.');
	}

	lineVertexPositionBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, lineVertexPositionBuffer);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(this.vertices), gl.STATIC_DRAW);

	lineVertexPositionBuffer.itemSize = 3;
	vert_count = this.vertices.length / 3;
	lineVertexPositionBuffer.numItems = vert_count;

	// specifying colors for line vertices
	lineVertexColorBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, lineVertexColorBuffer);
	
	//create the color array 
	var colors = [];
	for (var i=0; i < vert_count; i++) {
            colors = colors.concat(lineColor);
        }

	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
	lineVertexColorBuffer.itemSize = 4;
	lineVertexColorBuffer.numItems = vert_count;
	return gl;
    }
    
    function drawScene() {
	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
	
	mat4.perspective(45, gl.viewportWidth / gl.viewportHeight,
			 0.1, 100.0, pMatrix);
	
	mat4.identity(mvMatrix);
	mat4.translate(mvMatrix, [0.0, -0.5, zDist]);
	mvPushMatrix();
	mat4.rotate(mvMatrix, degToRad(xRot), [1, 0, 0]);
	mat4.rotate(mvMatrix, degToRad(yRot), [0, 1, 0]);
        //add the mouse rotation
        mat4.multiply(mvMatrix, mouseRotationMatrix);

	gl.bindBuffer(gl.ARRAY_BUFFER, lineVertexPositionBuffer);
	gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, lineVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);

	gl.bindBuffer(gl.ARRAY_BUFFER, lineVertexColorBuffer);
	gl.vertexAttribPointer(shaderProgram.vertexColorAttribute, lineVertexColorBuffer.itemSize, gl.FLOAT, false, 0, 0);


	setMatrixUniforms();
	gl.drawArrays(gl.LINES, 0, lineVertexPositionBuffer.numItems);
	mvPopMatrix();
    }

    function animate() {
        var timeNow = new Date().getTime();
	if (lastTime != 0) {
            var elapsed = timeNow - lastTime;
	    if(!mouseDown && rotateBool){
		xRot += (xSpeed * elapsed) / 1000.0;
		yRot += (ySpeed * elapsed) / 1000.0;
	    }
	}
	lastTime = timeNow;
    }
    
    function handleKeyDown(event) {
	currentlyPressedKeys[event.keyCode] = true;
    }
    
    function handleKeyUp(event) {
	currentlyPressedKeys[event.keyCode] = false;
    }
    
    function handleKeys() {
	if (currentlyPressedKeys[33]) {
	    // Page Up
	    zSpeed -= 0.05;
	}
	if (currentlyPressedKeys[34]) {
	    // Page Down
	    zSpeed += 0.05;
	}
	if (currentlyPressedKeys[90]) {
	    // z key
	    zDist += 0.05;
	}
	if (currentlyPressedKeys[88]) {
	    // x key
	    zDist -= 0.05;
	}
	if (currentlyPressedKeys[37]) {
	// Left cursor key
	    ySpeed -= 1;
	}
	if (currentlyPressedKeys[39]) {
	    // Right cursor key
	    ySpeed += 1;
	}
	if (currentlyPressedKeys[38]) {
	// Up cursor key
	    xSpeed -= 1;
	}
	if (currentlyPressedKeys[40]) {
	    // Down cursor key
	    xSpeed += 1;
	}
    }
    
    function tick() {
	requestAnimFrame(tick);
	drawScene();
	handleKeys();
	animate();
    }
    
   this.webGLStart = function(canvas_name) {
	var canvas = document.getElementById(canvas_name);
	gl_context = initGL(canvas);

	gl_context = initShaders();
	gl_context = this.initBuffers();
	
	gl_context.clearColor(0.0, 0.0, 0.0, 1.0);
	gl_context.enable(gl_context.DEPTH_TEST);
	
	canvas.onkeydown = handleKeyDown;
	canvas.onkeyup = handleKeyUp;
	canvas.onmousedown = handleMouseDown;
        canvas.onmouseup = handleMouseUp;
        canvas.onmousemove = handleMouseMove;
	tick();
    }
    this.webGLStart(this.canvasId)
}//end of glObject class
