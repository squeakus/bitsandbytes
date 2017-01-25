<?PHP

function getPostSQL($sectionCode) {

	if ($sectionCode == "secWP") {
		$sql = "INSERT INTO wpposts(threadID, memberID, threadTopic, postText, datePosted) VALUES ";
	}
	elseif ($sectionCode == "secWD") {
		$sql = "INSERT INTO wdposts(threadID, memberID, threadTopic, postText, datePosted) VALUES ";
	}
	elseif ($sectionCode == "secVB") {
		$sql = "INSERT INTO vbposts(threadID, memberID, threadTopic, postText, datePosted) VALUES ";
	}
	elseif ($sectionCode == "secXL") {
		$sql = "INSERT INTO xlposts(threadID, memberID, threadTopic, postText, datePosted) VALUES ";
	}
	elseif ($sectionCode == "secPH") {
		$sql = "INSERT INTO phposts(threadID, memberID, threadTopic, postText, datePosted) VALUES ";
	}
	else {
		$sql ="";
	}

	return $sql;

}

function getPostTable($sectionCode) {

	if ($sectionCode == "secWP") {
		$sql = "SELECT * FROM wpposts";
	}
	elseif ($sectionCode == "secWD") {
		$sql = "SELECT * FROM wdposts";
	}
	elseif ($sectionCode == "secVB") {
		$sql = "SELECT * FROM vbposts";
	}
	elseif ($sectionCode == "secXL") {
		$sql = "SELECT * FROM xlposts";
	}
	elseif ($sectionCode == "secPH") {
		$sql = "SELECT * FROM phposts";
	}
	else {
		$sql ="";
	}

	return $sql;

}



if ($_SERVER['REQUEST_METHOD'] == 'POST') {

		$secID = $_POST['h1'];
		$memID = $_POST['h2'];
		$posTopic = $_POST['tp'];
		$posText = $_POST['post'];

$posTopic = "'" . addslashes($posTopic) . "'";
$posText = "'" . addslashes($posText) . "'";
$memID = "'" . $memID . "'";

$date_today = date("Y-m-d H:i:s");
$date_today = "'" . $date_today . "'";

	$user_name = "root";
	$password = "";
	$database = "dbforum";
	$server = "127.0.0.1";

	$db_handle = mysql_connect($server, $user_name, $password);
	$db_found = mysql_select_db($database, $db_handle);

	if ($db_found) {

	//===================================================
	//		GET THE LAST POST
	//===================================================

		$SQL = getPostTable($secID);

		$result = mysql_query($SQL);
		$numRows = mysql_num_rows($result);

		//===================================================
		//	SET UP AN ARRAY TO HOLD THE threadID DATA
		//===================================================
		$posNums = array();

		//===================================================
		//	LOOP ROUND THE TABLE RESULTS AND GET THE threadID DATA ($pID = $row[0])
		//===================================================

		for ($i = 0; $i < $numRows; $i++) {
			$row = mysql_fetch_row($result);
			$pID = $row[0];

			//===================================================================================
			//	STRIP THE 'pos' FROM THE BEGINNING OF THE threadID AND PUT IT INTO THE ARRAY
			//===================================================================================
			$posNums[$i] = ltrim($pID, 'pos');

		}

		//===============================================================================
		//	SORT THE ARRAY, WITH THE LOWEST NUMBER FIRST AND HIGHEST NUMBER LAST
		//===============================================================================
		sort($posNums);

		//===================================================
		//	GET THE HIGHEST NUMBER FROM THE ARRAY
		//===================================================
		$lastID = end($posNums);

		//===================================================
		//	INCREMENT THE $lastID VARIABLE
		//===================================================
		$lastID++;

		//===================================================
		//	PUT THE pos BACK AT THE BEGINNING
		//===================================================
		$threadid = 'pos' . $lastID;
		$threadid = "'" . $threadid . "'";


		//===================================================
		//		GET THE SQL AND INSERT INTO TABLE
		//===================================================

		$tableSQL = getPostSQL($secID);
		$tableSQL = $tableSQL  . "(" . $threadid . "," . $memID . "," . $posTopic . "," .$posText . "," . $date_today . ")";


		$result = mysql_query($tableSQL);
		mysql_close($db_handle);

		if ($result) {
			print "Your Post has been added to the Forum." . "<BR>";
			print "<A HREF = forumTest.php>Back to the forum</A>" . "<BR>";
		}
		else {
			print "Already Posted" . "<BR>";
			print "thread id = " . $threadid . " num posts = " . $numRows . "<BR>";
			print $tableSQL;
		}
	}

}


?>


