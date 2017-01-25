function say(word) {
	  console.log("saying word:"+word);
}

function execute(someFunction, value) {
	  someFunction(value);
}

execute(say, "Hello");
