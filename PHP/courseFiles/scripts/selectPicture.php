<html>
 <head>
  <title>PHP Test</title>
 </head>
 <body>

<form Name = "f1" action="selectPicture.php" method="post">
	<select name="picture">
		<option value="none">Select a Picture</option>
		<option value="church">Church</option>
		<option value="kitten">Kitten</option>
		<option value="planet">Planet</option>
		<option value="cartoon">Cartoon</option>
		<option value="space">Space Image</option>
		<option value="abstract">Photoshop Abstract</option>
	</select>

	<input type="submit" name = "Submit" Value = "Choose an Image">
</form>

<?php 

if (isset($_POST['Submit'])) {

$picture = $_POST['picture'];

if ($picture == "church") {
	print ("<IMG SRC =images/church.jpg>");
}
else if ($picture == "kitten"){	
	print ("<IMG SRC =images/kitten.jpg>");
}
else if ($picture == "planet"){	
	print ("<IMG SRC =images/planet.jpg>");
}
else if ($picture == "cartoon"){	
	print ("<IMG SRC =images/cartoon.jpg>");
}
else if ($picture == "space"){	
	print ("<IMG SRC =images/stellar.jpg>");
}
else if ($picture == "abstract"){	
	print ("<IMG SRC =images/abstract.jpg>");
}
else {	
	print ("No Image to Display");
}

}


?>
</body>
</html>