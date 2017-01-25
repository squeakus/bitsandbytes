<?PHP
session_start();
if (!(isset($_SESSION['login']) && $_SESSION['login'] != '')) {
	header ("Location: login.php");
}
else {
	$memberid = $_SESSION['memID'];
}


if ($_SERVER['REQUEST_METHOD'] == 'GET') {
	$secCode = '';
	$postID = '';
	if (isset($_GET['pid'])) {
		$postID = $_GET['pid'];
		$secCode = $_GET['sec'];
	}
}

if ($secCode <> '') {

	$paragraph = "<P>";
	$startForm = "<FORM NAME ='form2' METHOD ='POST' ACTION ='results.php'>";

	$tbLabel = "<B>Type your reply:</B> <BR>";
	$textArea = "<TEXTAREA Name = post Rows = 15 Cols = 40>Some text here</TEXTAREA>";

	$hidSec = "<INPUT TYPE = Hidden Name = h1  VALUE =" . $secCode . ">";
	$hidPost = "<INPUT TYPE = Hidden Name = h2  VALUE =" . $postID . ">";
	$hidMem = "<INPUT TYPE = Hidden Name = h3  VALUE =" . $memberid . ">";


	$formButton = "<INPUT TYPE = 'Submit' Name = 'Submit2'  VALUE = 'Add your reply'>";

	$endForm = "</FORM>";

	print $startForm;
	print $tbLabel;

		print $textArea;
		print $hidSec;
		print $hidPost;
		print $hidMem;
		print$paragraph;
		print $formButton;

	print $endForm;

}
else {
	print "forum not available";
	print $memberid . "<BR>";
}




?>