/* Read a file  using xmlhttprequest 

If the HTML file with your javascript app has been saved to disk, 
this is an easy way to read in a data file.  Writing out is 
more complicated and requires either an ActiveX object (IE) 
or XPCOM (Mozilla).

fname - relative path to the file
callback - function to call with file text
*/

function readFileHttp(fname, callback) {
   xmlhttp = getXmlHttp();
   xmlhttp.onreadystatechange = function() {
      if (xmlhttp.readyState==4) { 
          callback(xmlhttp.responseText); 
      }
   }
   xmlhttp.open("GET", fname, true);
   xmlhttp.send(null);
}


function setText(text){
    document.getElementById('byte_content').textContent = text;
}

/*
Return a cross-browser xmlhttp request object
*/

function getXmlHttp() {
   if (window.XMLHttpRequest) {
      xmlhttp=new XMLHttpRequest();
   } else if (window.ActiveXObject) {
      xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
   }
   if (xmlhttp == null) {
      alert("Your browser does not support XMLHTTP.");
   }
   return xmlhttp;
}


function initListeners(){
    console.log("button listeners initialised");
    document.querySelector('.readButton').addEventListener('click', function(evt) {
	if (evt.target.tagName.toLowerCase() == 'button'){ 
	    readFileHttp('test.txt', setText)
	}
    }, false);
}

