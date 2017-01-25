<html>
<head>
<title>A BASIC HTML FORM</title>

<?PHP
$username ="default";


if (isset($_POST['Submit1'])) {

	$username = $_POST['username'];

	print $username;

}
else {
	$username ="";

}


?>

</head>
<body>

<FORM NAME ="form1" METHOD ="POST" ACTION ="magicTest.php">



<INPUT TYPE = 'TEXT' Name ='username'  value="<?PHP print $username ; ?>">
<INPUT TYPE = "Submit" Name = "Submit1"  VALUE = "Login">
</FORM>






</body>
</html>

