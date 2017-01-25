<?PHP

$uname = "";
$pword = "";
$errorMessage = "";
//==========================================
//	ESCAPE DANGEROUS SQL CHARACTERS
//==========================================
function quote_smart($value, $handle) {

   if (get_magic_quotes_gpc()) {
       $value = stripslashes($value);
   }

   if (!is_numeric($value)) {
       $value = "'" . mysql_real_escape_string($value, $handle) . "'";
   }
   return $value;
}

if ($_SERVER['REQUEST_METHOD'] == 'POST'){
	$uname = $_POST['username'];
	$pword = $_POST['password'];

	$uname = htmlspecialchars($uname);
	$pword = htmlspecialchars($pword);

	//==========================================
	//	CONNECT TO THE LOCAL DATABASE
	//==========================================
	$user_name = "root";
	$pass_word = "";
	$database = "dbforum";
	$server = "127.0.0.1";

	$db_handle = mysql_connect($server, $user_name, $pass_word);
	$db_found = mysql_select_db($database, $db_handle);

	if ($db_found) {

		//$uname = quote_smart($uname, $db_handle);
		//$pword = quote_smart($pword, $db_handle);

		$SQL = "SELECT  * FROM members WHERE username = '$uname' AND password = '$pword'";

		$result = mysql_query($SQL, $db_handle);

		if ($result) {
			$num_rows = mysql_num_rows($result);

			if ($num_rows > 0) {

				$db_field = mysql_fetch_assoc($result);
				$mem = $db_field['memberID'];
				session_start();
				$_SESSION['login'] = "1";
				$_SESSION['memID'] = $mem;

				header ("Location: forumTest.php");
			}
			else {
				$errorMessage = "Invalid Login.";

				session_start();
				$_SESSION['login'] = '';

				//==========================================
				//	YOUR SIGNUP PAGE HERE
				//==========================================
				//header ("Location: signup.php");
			}	
		}
		else {
			$errorMessage = "Error logging on - no results";
		}


	}

	else {
		$errorMessage = "Error logging on - last error";
	}

}


?>


<html>
<head>
<title>Basic Login Script</title>
</head>
<body>

<FORM NAME ="form1" METHOD ="POST" ACTION ="login.php">

Username: <INPUT TYPE = 'TEXT' Name ='username'  value="<?PHP print $uname;?>" maxlength="20">
Password: <INPUT TYPE = 'TEXT' Name ='password'  value="<?PHP print $pword;?>" maxlength="16">

<P align = center>
<INPUT TYPE = "Submit" Name = "Submit1"  VALUE = "Login">
</P>

</FORM>

<P>
<?PHP print $errorMessage;?>




</body>
</html>