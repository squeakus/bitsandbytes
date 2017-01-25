<?PHP

//================----EXERCISE----==================================
//	FORUM POSTS ARE DISPLAYED BY DATE POSTED. 
//	TRY CHANGING DESC TO ASC AND SEE WHAT HAPPENS.
//================----END EXERCISE----==================================

session_start();
if (!(isset($_SESSION['login']) && $_SESSION['login'] != '')) {
	$nonMember = "YOU NEED TO BE LOGGED IN TO POST (MAKE SURE COOKIES ARE ENABLED IN YOUR BROWSER)";
}
else {
	$nonMember = '';
}

include 'forumHTML.php';

//==================================================
//	GET THE SQL FOR THE POSTS	
//==================================================
function getPostSQL($sectionCode) {

	if ($sectionCode == "secWP") {
		$sql = "SELECT * FROM wpposts ORDER BY datePosted DESC";
	}
	elseif ($sectionCode == "secWD") {
		$sql = "SELECT * FROM wdposts ORDER BY datePosted DESC";
	}
	elseif ($sectionCode == "secVB") {
		$sql = "SELECT * FROM vbposts ORDER BY datePosted DESC";
	}
	elseif ($sectionCode == "secXL") {
		$sql = "SELECT * FROM xlposts ORDER BY datePosted DESC";
	}
	elseif ($sectionCode == "secPH") {
		$sql = "SELECT * FROM phposts ORDER BY datePosted DESC";
	}
	else {
		$sql ="";
	}

	return $sql;

}

function getReplySQL($sectionCode, $reply) {

	if ($sectionCode == "secWP") {
		$sql = "SELECT * from wpreplies WHERE wpreplies.threadID = '$reply'";
	}
	elseif ($sectionCode == "secWD") {
		$sql = "SELECT * from wdreplies WHERE wpreplies.threadID = '$reply'";
	}
	elseif ($sectionCode == "secVB") {
		$sql = "SELECT * from vbreplies WHERE wpreplies.threadID = '$reply'";
	}
	elseif ($sectionCode == "secXL") {
		$sql = "SELECT * from xlreplies WHERE wpreplies.threadID = '$reply'";
	}
	elseif ($sectionCode == "secPH") {
		$sql = "SELECT * from phreplies WHERE wpreplies.threadID = '$reply'";
	}
	else {
		$sql ="";
	}

	return $sql;

}


if ($_SERVER['REQUEST_METHOD'] == 'GET') {
	$secCode = '';
	if (isset($_GET['sID'])) {
		$secCode = $_GET['sID'];
	}
}

if ($secCode <> '') {
$postData[] = array();

$replyHTML = "?sid=" . $secCode;
$replyHTML = "<A HREF = postForm.php" . $replyHTML . ">Create a new post</A>";


$forum = $secCode;

	$user_name = "root";
	$password = "";
	$database = "dbforum";
	$server = "127.0.0.1";

	$db_handle = mysql_connect($server, $user_name, $password);
	$db_found = mysql_select_db($database, $db_handle);


	if ($db_found) {

		//================================================================
		//	SET UP A 2D ARRAY TO HOLD THE DATA FROM THE POSTS TABLE	
		//===============================================================

		$secCode = getPostSQL($secCode);
		$result = mysql_query($secCode);
		$totalRows = 0;
		$totalRows = mysql_num_rows($result);

if ($totalRows <> 0) {

		for ($i = 0; $i < $totalRows; ++$i) {
  			$postData[$i] = mysql_fetch_array($result);

			//==========================================
			//		EXERCISE CODE
			//==========================================
			//print $postData[$i]['threadID'] . " " . "<BR>";
			//print $postData[$i]['memberID'] . " ";
			//print $postData[$i]['threadTopic'] . " ";
			//print $postData[$i]['postText'] . " ";
			//print "NEW ARRAY DATA:" . $postData[$i]['numRows'] . " ";
			//print $postData[$i]['datePosted'] . "<BR>";
		}

			//==========================================
			//		EXERCISE CODE
			//==========================================
			//print $postData[0]['threadTopic'] . "<BR>";

			//for ($i = 0; $i < $totalRows; ++$i) {
				//$postData[$i]['newPosition'] = $i;
			//}

			//print $postData[1]['newPosition'] . "<BR>";


		//================================================================
		//	FIND OUT HOW MANY REPLIES THERE ARE FOR EACH POST
		//	ADD A NEW VALUE TO THE 2D ARRAY
		//===============================================================		
		$cnt = count($postData);

		for ($i = 0; $i < $cnt; ++$i) {

			$rep = $postData[$i]['threadID'];

			$repSQL = getReplySQL($forum, $rep);
			$result = mysql_query($repSQL);
			$numRows = mysql_num_rows($result);

			$postData[$i]['numRows'] = $numRows;
		}	


		//================================================================
		//	FIND OUT WHICH MEMBER POSTED, AND ADD IT THE 2D ARRAY
		//===============================================================
		for ($i = 0; $i < $cnt; ++$i) {

			$memb = $postData[$i]['memberID'];

			$memSQL = "SELECT * from members WHERE memberID = '$memb'";
			$result2 = mysql_query($memSQL);

			if ($result2) {
				$db_field = mysql_fetch_assoc($result2);
				$memName = $db_field['username'];
				$postData[$i]['member'] = $memName;
			}
		}


		mysql_close($db_handle);
		//=====================================
		//	PRINT THE TABLE OUT
		//=====================================

		print "<CENTER>";
		print $TableStart;
		print $tableHeaders;

		for ($i = 0; $i < $cnt; ++$i) {	

			print $RowStart;

			print $tdStart . $postData[$i]['member'] . $tdEnd;
print $tdStart . $hrefStart . "=" . $postData[$i]['threadID'] . "&forum=" . $forum . "&pageID=0" . ">" . $postData[$i]['threadTopic'] . $hrefEnd . $tdEnd;
			print $tdStart . $postData[$i]['numRows'] . $tdEnd;

			print $RowEnd;
		}

			print $TableEnd;
			print "</CENTER>";


		//========================================================
		//	DISPLAY A LINK SO THAT MEMBERS CAN POST A REPLY
		//	IF NOT A MEMBER, DISPLAY A MESSAGE AND LOGON LINK
		//=========================================================

		if ($nonMember == '') {
			print "<P align = center>" . $replyHTML . "</P>";
		}
		else {
			print "<P align = center>" . $nonMember . "</P>";
			print "<P align = center>" . "<A HREF = login.php>Login Here</A>" . "</P>";
		}


}
elseif ($totalRows == 0) {
	print "This Forum is not yet available";
}

	}

}
else {
	print "Forum Not Available";
}

?>











