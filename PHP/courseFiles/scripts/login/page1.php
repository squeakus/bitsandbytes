<?PHP
session_start();
if (!(isset($_SESSION['login']) && $_SESSION['login'] != '')) {
	header ("Location: login3.php");
}

?>

	<html>
	<head>
	<title>Basic Login Script</title>


	</head>
	<body>




	User Logged in
<P>
<A HREF = page2.php>Log out</A>

	</body>
	</html>
