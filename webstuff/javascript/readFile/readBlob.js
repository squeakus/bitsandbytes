

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


function initListeners(){
    console.log("listeners initialised");
    document.querySelector('.readButton').addEventListener('click', function(evt) {
	if (evt.target.tagName.toLowerCase() == 'button') {
	    var startByte = evt.target.getAttribute('data-startbyte');
	    var endByte = evt.target.getAttribute('data-endbyte');
	    readBlob(startByte, endByte);
	}
    }, false);
}

