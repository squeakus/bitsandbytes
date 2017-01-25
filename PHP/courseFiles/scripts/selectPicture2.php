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

	Switch ($picture) {
		Case 'kitten':
			print ("<IMG SRC =images/kitten.jpg>");
			break;

		Case 'church':
			print ("<IMG SRC =images/church.jpg>");
			break;

		Case 'planet':
			print ("<IMG SRC =images/planet.jpg>");
			break;
		Case 'cartoon':
			print ("<IMG SRC =images/cartoon.jpg>");
			break;
		Case 'space':
			print ("<IMG SRC =images/stellar.jpg>");
			break;
		Case 'abstract':
			print ("<IMG SRC =images/abstract.jpg>");
			break;
		default:
			print ("No Image Selected");
	}
}


?>
</body>
</html>