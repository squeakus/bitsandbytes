var gl;
var vertices;

function initGL(canvas) {
    try {
        gl = canvas.getContext("experimental-webgl");
        gl.viewportWidth = canvas.width;
        gl.viewportHeight = canvas.height;
    } catch (e) {
        }
        if (!gl) {
            alert("Could not initialise WebGL, sorry :-(");
        }
    }

function readBlob(opt_startByte, opt_stopByte) {
    var files = document.getElementById('files').files;
    if (!files.length) {
	alert('Please select a file!');
	return;
    }
    
    var file = files[0];
    var start = parseInt(opt_startByte) || 0;
    var stop = parseInt(opt_stopByte) || file.size - 1;
    
    var reader = new FileReader();
    
    // If we use onloadend, we need to check the readyState.
    reader.onloadend = function(evt) {
	if (evt.target.readyState == FileReader.DONE) { // DONE == 2
            document.getElementById('byte_content').textContent = evt.target.result;
	    var mesh_string = evt.target.result;
	    mesh_string = someText = mesh_string.replace(/(\r\n|\n|\r)/gm,"");
	    vertices = mesh_string.split(",");
	}
    };
    
    if (file.webkitSlice) {
	var blob = file.webkitSlice(start, stop + 1);
    } else if (file.mozSlice) {
	var blob = file.mozSlice(start, stop + 1);
    }
    reader.readAsBinaryString(blob);
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

var shaderProgram;

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
    
    shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
    gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);
    
    shaderProgram.pMatrixUniform = gl.getUniformLocation(shaderProgram, "uPMatrix");
    shaderProgram.mvMatrixUniform = gl.getUniformLocation(shaderProgram, "uMVMatrix");
}

var mvMatrix = mat4.create();
var pMatrix = mat4.create();
var mvMatrixStack = [];

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

var triangleVertexPositionBuffer;
var squareVertexPositionBuffer;

function initBuffers() {
    lineVertexPositionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, lineVertexPositionBuffer);
    
    if (window.File && window.FileReader && window.FileList && window.Blob){
	console.log("success!");
    } else {
	alert('The File APIs are not fully supported in this browser.');
    }
    
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    lineVertexPositionBuffer.itemSize = 3;
    vert_count = vertices.length / 3
    console.log("length"+vert_count);
    lineVertexPositionBuffer.numItems = vert_count;
}

var rTri = 0;
var rLine = 0;


function drawScene() {
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
    mat4.perspective(45, gl.viewportWidth / gl.viewportHeight,
		     0.1, 100.0, pMatrix);
    
    mat4.identity(mvMatrix);
    mat4.translate(mvMatrix, [0.0, -0.5, -1.5]);
    
    mvPushMatrix();
    mat4.rotate(mvMatrix, degToRad(rLine), [0, 1, 0]);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineVertexPositionBuffer);
    gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, lineVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
    setMatrixUniforms();
    gl.drawArrays(gl.LINES, 0, lineVertexPositionBuffer.numItems);
    mvPopMatrix();
}

var lastTime = 0;

function animate() {
        var timeNow = new Date().getTime();
    if (lastTime != 0) {
        var elapsed = timeNow - lastTime;
            rLine += (100 * elapsed) / 1000.0;
    }
    lastTime = timeNow;
}


function tick() {
    requestAnimFrame(tick);
    drawScene();
    animate();
}

function webGLStart() {
    var canvas = document.getElementById("lesson01-canvas");
    initGL(canvas);
    initShaders();
    initBuffers();
    
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.enable(gl.DEPTH_TEST);
    tick(); 
}

function initListeners(){
    document.querySelector('.readBytesButtons').addEventListener('click', function(evt) {
	if (evt.target.tagName.toLowerCase() == 'button') {
	    var startByte = evt.target.getAttribute('data-startbyte');
	    var endByte = evt.target.getAttribute('data-endbyte');
	    readBlob(startByte, endByte);
	}
    }, false);

    document.querySelector('.renderButton').addEventListener('click', function(evt) {
	if (evt.target.tagName.toLowerCase() == 'button') {
	    webGLStart();
	}
    }, false);
}

initListeners();