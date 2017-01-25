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

	if (isset($_GET['sid'])) {
		$secCode = $_GET['sid'];
	}
}

if ($secCode <> '') {

	$paragraph = "<P>";
	$startForm = "<FORM NAME ='form2' METHOD ='POST' ACTION ='resultsP.php'>";

	$tbLabel2 = "<B>Post Topic:</B> <BR>";
	$tbPostTopic= "<INPUT TYPE = Text Name = tp  VALUE = 'Post Topic Here' SIZE = 50><P>";


	$tbLabel = "<B>Type your Post:</B> <BR>";
	$textArea = "<TEXTAREA Name = post Rows = 15 Cols = 37>Post Text here</TEXTAREA>";

	$hidSec = "<INPUT TYPE = Hidden Name = h1  VALUE =" . $secCode . ">";
	$hidMem = "<INPUT TYPE = Hidden Name = h2  VALUE =" . $memberid . ">";


	$formButton = "<INPUT TYPE = 'Submit' Name = 'Submit2'  VALUE = '  Post  '>";

	$endForm = "</FORM>";

	print $startForm;
	print $tbLabel2;
	print $tbPostTopic;
	print $tbLabel;

		print $textArea;
		print $hidSec;
		print $hidMem;
		print$paragraph;
		print $formButton;

	print $endForm;

}
else {
	print "forum not available";
}




?>