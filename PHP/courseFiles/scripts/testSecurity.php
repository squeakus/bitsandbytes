<html>
 <head>
  <title>Test Attack</title>


<?php 

if ($_SERVER['REQUEST_METHOD'] == 'POST'){
	$first_name = $_POST['first_name'];

//$first_name = htmlspecialchars($first_name);
$first_name = strip_tags($first_name, "<B>");
//$first_name = htmlentities($first_name);

	echo $first_name;
}
?>


 </head>
 <body>


</body>

<FORM Method = "POST" action ="testSecurity.php">
	

Add your comments<BR>
<TEXTAREA Rows =7 Cols = 20 NAME = "first_name">test message</TEXTAREA>
<P>	
<input type="submit" name="Submit" value="Submit">
</FORM>

<P>

</html>
