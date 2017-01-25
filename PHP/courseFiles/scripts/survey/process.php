<?PHP
session_start();
if ((isset($_SESSION['hasVoted']))) {
	if ($_SESSION['hasVoted'] = '1') {
		print "You've already voted";
	}
}
else {
	if (isset($_GET['Submit1']) && isset($_GET['q'])) {

		$selected_radio = $_GET['q'];

		$user_name = "root";
		$password = "";
		$database = "surveyTest";
		$server = "127.0.0.1";

		$db_handle = mysql_connect($server, $user_name, $password);
		$db_found = mysql_select_db($database, $db_handle);

		if ($db_found) {
			$_SESSION['hasVoted'] = '1';
			$SQL = "UPDATE answers SET $selected_radio = $selected_radio + 1";
			$result = mysql_query($SQL);
			mysql_close($db_handle);
			print "Thanks for voting!";
		}
		else {
		print "database error";
		}
	}
	else {
		print "You didn't selected a voting option!";
	}
}

?>

<html>
<head>
<title>Process Survey</title>
</head>



<body>

</body>
</html>

