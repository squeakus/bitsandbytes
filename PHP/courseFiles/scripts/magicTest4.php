<html>
<head>
<title>A BASIC HTML FORM</title>

<?PHP

//=======THIS IS THE FUNCTION FROM THE PHP MANUAL
function quote_smart($value, $handle)
{
   // Stripslashes
   if (get_magic_quotes_gpc()) {
       $value = stripslashes($value);
   }
   // Quote if not integer
   if (!is_numeric($value)) {
       $value = "'" . mysql_real_escape_string($value, $handle) . "'";
   }
   return $value;
}

//=======END OF FUNCTION FROM THE PHP MANUAL


$username ="default";
$password ="";
$email = "";

if (isset($_POST['Submit1'])) {

	$username = htmlspecialchars($_POST['username']);
	$password = htmlspecialchars($_POST['password']);
	$email = htmlspecialchars($_POST['email']);


	$user_name = "root";
	$pass_word = "";
	$database = "membertest";
	$server = "127.0.0.1";

	$db_handle = mysql_connect($server, $user_name, $pass_word);
	$db_found = mysql_select_db($database, $db_handle);

if ($db_found) {
	
	//=====THE NEW FUNCTION IS BEING CALLED HERE. ONLY THE $email VARIABLE IS BEING CHECKED
	//=====$password = quote_smart($password, $db_handle);
	//=====$username = quote_smart($username, $db_handle);

	$email = quote_smart($email, $db_handle);

	$SQL = "SELECT * FROM members WHERE email = $email";
	$result = mysql_query($SQL);


//======CHECK TO SEE IF THE VARIABLE IS TRUE
	if ($result) {
		while ($db_field = mysql_fetch_assoc($result)) {

			print $db_field['ID'] . "<BR>";
			print $db_field['username'] . "<BR>";
			print $db_field['password'] . "<BR>";
			print $db_field['email'] . "<BR>";
		}

		mysql_close($db_handle);

	}

}
else {
	print "Database NOT Found ";
	mysql_close($db_handle);
}


}


?>


</head>
<body>

<FORM NAME ="form1" METHOD ="POST" ACTION ="magicTest4.php">



username <INPUT TYPE = 'TEXT' Name ='username'  value="<?PHP print $username ; ?>">
password <INPUT TYPE = 'TEXT' Name ='password'  value="">
<P>
email address <INPUT TYPE = 'TEXT' Name ='email'  value="<?PHP print $email ; ?>">


<INPUT TYPE = "Submit" Name = "Submit1"  VALUE = "Login">
</FORM>






</body>
</html>

