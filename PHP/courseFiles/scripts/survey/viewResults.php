<?PHP
$question = '';
$answerA = '';
$answerB = '';
$answerC = '';

$imgTagA = '';
$imgWidthA = '0';

$imgTagB = '';
$imgWidthB = '0';

$imgTagC = '';
$imgWidthC = '0';

$imgHeight = '10';
$totalP = '';
$percentA = '0';
$percentB = '0';
$percentC = '0';

$qA = '';
$qB = '';
$qC = '';

if (isset($_GET['Submit2'])) {

		$qNum = $_GET['h1'];

		$user_name = "root";
		$password = "";
		$database = "surveyTest";
		$server = "127.0.0.1";

		$db_handle = mysql_connect($server, $user_name, $password);
		$db_found = mysql_select_db($database, $db_handle);

		if ($db_found) {

			$SQL = "SELECT * FROM tblquestions, answers WHERE tblquestions.QID = answers.QID AND answers.QID = '$qNum'";
			$result = mysql_query($SQL);
			$db_field = mysql_fetch_assoc($result);

			$question = $db_field['Question'];
			$answerA = $db_field['A'];
			$answerB = $db_field['B'];
			$answerC = $db_field['C'];

			$qA = $db_field['qA'];
			$qB = $db_field['qB'];
			$qC = $db_field['qC'];

			$imgWidthA = $answerA;
			$imgWidthB = $answerB;
			$imgWidthC = $answerC;

$totalP = $answerA + $answerB + $answerC;

$percentA = (($answerA * 100) / $totalP);
$percentA = floor($percentA);

$percentB = (($answerB * 100) / $totalP);
$percentB = floor($percentB);

$percentC = (($answerC * 100) / $totalP);
$percentC = floor($percentC);

$imgWidthA = $percentA * 2;
$imgWidthB = $percentB * 2;
$imgWidthC = $percentC * 2;


			$imgTagA = "<IMG SRC = 'red.jpg' Height = " . $imgHeight . " WIDTH = " . $imgWidthA. ">";
			$imgTagB = "<IMG SRC = 'red.jpg' Height = " . $imgHeight . " WIDTH = " . $imgWidthB . ">";
			$imgTagC = "<IMG SRC = 'red.jpg' Height = " . $imgHeight . " WIDTH = " . $imgWidthC . ">";

			mysql_close($db_handle);


		}
		else {
			print "database error";
		}


}
else {
	print "no results to display";
}
?>

<html>
<head>
<title>Process Survey</title>
</head>



<body>

<?PHP
print $question . "<BR>";
print $imgTagA . " " . $percentA . "% " . $qA . "<BR>";
print $imgTagB . " " . $percentB . "% " . $qB . "<BR>";
print $imgTagC . " " . $percentC . "% " . $qC . "<BR>";

?>
</body>
</html>