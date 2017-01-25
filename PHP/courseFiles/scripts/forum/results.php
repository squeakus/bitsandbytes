<?PHP

function getReplySQL($sectionCode) {

	if ($sectionCode == "secWP") {
		$sql = "INSERT INTO wpreplies (threadID, memberID, reply, dateReplied) VALUES ";
	}
	elseif ($sectionCode == "secWD") {
		$sql = "INSERT INTO wdreplies (threadID, memberID, reply, dateReplied) VALUES ";
	}
	elseif ($sectionCode == "secVB") {
		$sql = "INSERT INTO vbreplies (threadID, memberID, reply, dateReplied) VALUES ";
	}
	elseif ($sectionCode == "secXL") {
		$sql = "INSERT INTO xlreplies (threadID, memberID, reply, dateReplied) VALUES ";
	}
	elseif ($sectionCode == "secPH") {
		$sql = "INSERT INTO phreplies (threadID, memberID, reply, dateReplied) VALUES ";
	}
	else {
		$sql ="";
	}

	return $sql;

}

if ($_SERVER['REQUEST_METHOD'] == 'POST') {

		$secID = $_POST['h1'];
		$posID = $_POST['h2'];
		$memID = $_POST['h3'];
		$repText = $_POST['post'];

$strCount = strlen($repText);
if ($strCount > 255) {
print "too many characters in your reply";

}
else {
$repText = "'" . $repText . "'";
$posID = "'" . $posID  . "'";
$memID = "'" . $memID . "'";


$date_today = date("Y-m-d H:i:s");
$date_today = "'" . $date_today . "'";


$tableSQL = getReplySQL($secID);
$tableSQL = $tableSQL  . "(" . $posID . "," . $memID . "," . $repText . "," . $date_today . ")";


	$user_name = "root";
	$password = "";
	$database = "dbforum";
	$server = "127.0.0.1";

	$db_handle = mysql_connect($server, $user_name, $password);
	$db_found = mysql_select_db($database, $db_handle);

	if ($db_found) {
		$result = mysql_query($tableSQL);
		mysql_close($db_handle);

		if ($result) {
			print "Your Reply has been added to the Forum." . "<BR>";
			print "<A HREF = forumTest.php>Back to the forum</A>" . "<BR>";
		}
		else {
			print "no results" . "<BR>";
		}
	}

}


}

?>







