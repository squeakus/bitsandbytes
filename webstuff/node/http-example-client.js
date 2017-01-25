var http = require("http")
  , util = require("sys") // 0.3.x: require("util")
  , fs = require("fs")
  , client = http.createClient(80, "example.com")
  , request = client.request("POST", "/", {"host":"example.com"})
 
// send body chunks
request.write("hello, world")
 
// pump a file through
// You'd normally have to call request.end() to actually
// have it send out, but the pump will .end() when the file is done.
// 0.3.x: fs.createReadStream(__filename).pipe(request)
util.pump(fs.createReadStream(__filename), request)
 
 
request.on("response", function (response) {
  // got a response
  console.log("response: "+response.statusCode)
  console.log(util.inspect(response.headers))
  // read the body
  // could listen to "data" and "end" manually, or just pump somewhere
  // pumping *into* stdout is kinda lame, because it closes it at the end.
  util.pump(response, process.stdout)
 
  // this is how to buffer it. not usually the right thing to do.
  // note that an array of buffers is used rather than one big string,
  // so that we don't get bitten by multibyte chars on the boundaries,
  // or take the performance hit of copying the data to/from v8's heap twice.
  // (once to put it into the string, then to get it out later)
  var bodyParts = []
    , bytes = 0
  response.on("data", function (c) {
    bodyParts.push(c)
    bytes += c.length
  })
  response.on("end", function () {
    // flatten into one big buffer
    // it'd be cooler if fs.writeFile et al could take an array
    // of buffers and use writev, but alas, not at this time.
    console.error(bytes, typeof bytes)
    var body = new Buffer(bytes)
      , copied = 0
    bodyParts.forEach(function (b) {
      b.copy(body, copied, 0)
      copied += b.length
    })
    fs.writeFile("the-response", body, function (er) {
      if (er) {
        console.error(er.stack || er.message)
        return process.exit(1)
      }
      console.error("ok")
    })
  })
 })
