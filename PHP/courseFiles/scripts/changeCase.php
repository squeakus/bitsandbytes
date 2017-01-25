<html>
<head>
<title>Change Case</title>

<?PHP
	$full_name = 'bill gates';

	if (isset($_POST['Submit1'])) {
	

		$full_name = $_POST['username'];

		$full_name = ucwords($full_name);
	}
?>

</head>
<body>

<FORM NAME ="form1" METHOD ="POST" ACTION ="changeCase.php">
	<INPUT TYPE = 'TEXT' Name ='username'  value="<?PHP print $full_name; ?>" >
	<INPUT TYPE = "Submit" Name = "Submit1"  VALUE = "Login">
</FORM>



</body>
</html>

