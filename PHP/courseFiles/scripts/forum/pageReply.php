<?PHP

session_start();
if (!(isset($_SESSION['login']) && $_SESSION['login'] != '')) {
	$nonMember = "YOU NEED TO BE LOGGED IN TO POST (MAKE SURE COOKIES ARE ENABLED IN YOUR BROWSER)";
}
else {
	$nonMember = '';
}

include 'forumHTMLReply.php';
$numRows = 0;

//==================================================
//	GET THE SQL FOR THE POSTS	
//==================================================
function getReplySQL($sectionCode) {
	if ($sectionCode == "secWP") {
		$sql = "SELECT * FROM wpreplies WHERE threadID=";
	}
	elseif ($sectionCode == "secWD") {
		$sql = "SELECT * FROM wdreplies WHERE threadID = ";
	}
	elseif ($sectionCode == "secVB") {
		$sql = "SELECT * FROM vbreplies WHERE threadID = ";
	}
	elseif ($sectionCode == "secXL") {
		$sql = "SELECT * FROM xlreplies WHERE threadID = ";
	}
	elseif ($sectionCode == "secPH") {
		$sql = "SELECT * FROM phreplies WHERE threadID = ";
	}
	else {
		$sql ="";
	}

	return $sql;
}

function getPostSQL($sectionCode) {
	if ($sectionCode == "secWP") {
		$sql = "SELECT * FROM wpposts WHERE wpposts.threadID =";
	}
	elseif ($sectionCode == "secWD") {
		$sql = "SELECT * FROM wdposts WHERE wdposts.threadID =";
	}
	elseif ($sectionCode == "secVB") {
		$sql = "SELECT * FROM vbposts WHERE vbposts.threadID =";
	}
	elseif ($sectionCode == "secXL") {
		$sql = "SELECT * FROM xlposts WHERE xlposts.threadID =";
	}
	elseif ($sectionCode == "secPH") {
		$sql = "SELECT * FROM phposts WHERE phposts.threadID =";
	}
	else {
		$sql ="";
	}
	return $sql;
}


if ($_SERVER['REQUEST_METHOD'] == 'GET') {
	$secCode = '';
	$postID = '';
	if (isset($_GET['rID'])) {
		$postID = $_GET['rID'];
		$secCode = $_GET['forum'];
		$pageID = $_GET['pageID'];
	}
}

if ($secCode <> '') {

		//================================================================
		//	BUILD UP THE LINK SO THAT A MEMBER CAN REPLY
		//================================================================
		$replyHTML = "?pid=" . $postID . "&sec=" . $secCode;
		$replyHTML = "<A HREF = replyForm.php" . $replyHTML . ">Reply to this post</A>";

		//================================================================
		//	GET THE SQL FOR THE POSTER
		//================================================================
		$posCode = getPostSQL($secCode);
		$posCode = $posCode . "'" . $postID . "'";



		//================================================================
		//	OPEN A CONNECTION TO THE DATABASE
		//================================================================
		$user_name = "root";
		$password = "";
		$database = "dbforum";
		$server = "127.0.0.1";
		$db_handle = mysql_connect($server, $user_name, $password);
		$db_found = mysql_select_db($database, $db_handle);

	if ($db_found) {
		//================================================================
		//	GET THE SQL FOR ALL THE REPLIES
		//================================================================
		$repCode = getReplySQL($secCode);
		$repCode = $repCode  . "'" . $postID . "'";

		$result = mysql_query($repCode);
		$totalRows = mysql_num_rows($result);

if ($totalRows <> 0) {


		//================================================================
		//	GET SOME SQL TO LIMIT THE NUMBER OF ROWS FROM DATABASE
		//================================================================

		$repCode = getReplySQL($secCode);
		$repCode = $repCode  . "'" . $postID . "'" . " LIMIT " . $pageID . ", 10";


		//================================================================
		//	RETURN THE LIMITED ROWS AND PUT THEM INTO AN ARRAY
		//================================================================
		$result = mysql_query($repCode);
		$numRows = mysql_num_rows($result);
		if ($result) {
			for ($i = 0; $i < $numRows; ++$i) {
  				$replyData[$i] = mysql_fetch_array($result);
			}

		//================================================================
		//	FIND OUT HOW MANY LINKS ARE NEEDED
		//===============================================================
			$cnt = count($replyData);

			//$endNum = $totalRows;
			$linkNum = floor($totalRows / 10);

		}
		else {
			print "forum error";
		}


		//================================================================
		//	GET INFORMATION ABOUT THE THREAD
		//===============================================================
		$result = mysql_query($posCode);
		$numRows = mysql_num_rows($result);

		if ($numRows == 1) {
			$db_field = mysql_fetch_assoc($result);
			$topic = $db_field['threadTopic'];
			$postText = $db_field['postText'];
			$datePosted = $db_field['datePosted'];
			$memPost = $db_field['memberID'];
		}


		//================================================================
		//	FIND OUT WHICH MEMBER POSTED THE THREAD
		//===============================================================

			$memSQL = "SELECT * from members WHERE memberID = '$memPost'";
			$result = mysql_query($memSQL);
			if ($result) {
				$db_field = mysql_fetch_assoc($result);
				$postName = $db_field['username'];
			}


		//================================================================
		//	FIND OUT WHICH MEMBERS REPLIED
		//===============================================================

		for ($i = 0; $i < $cnt; ++$i) {

			$memb = $replyData[$i]['memberID'];
			$memSQL = "SELECT * from members WHERE memberID = '$memb'";
			$result2 = mysql_query($memSQL);

			if ($result2) {
				$db_field = mysql_fetch_assoc($result2);
				$memName = $db_field['username'];
				$replyData[$i]['member'] = $memName;
			}
		}


		mysql_close($db_handle);


		//=====================================
		//	PRINT THE LINKS OUT
		//=====================================
		print "<p ALIGN = CENTER>";
		$linkCount = 0;
		$pageCount = 1;
		for ($i = 0; $i <= $linkNum; ++$i) {
			$linkPages = "<A HREF = pageReply.php?rID=" . $postID . "&forum=" . $secCode;
			$linkPages = $linkPages . "&pageID=" . $linkCount . ">Page " . $pageCount . "</A>";
			print $linkPages . " ";
			$linkCount = $linkCount + 10;
			$pageCount++;
		}
		
		print "</p>";

		//=====================================
		//	PRINT THE TABLE OUT
		//=====================================
		print "<CENTER>";
		print "<B>" . $topic . "</B>";
		print $TableStart;
		print $tableHeaders2;

		print $RowStart;
		print $tdStart . $postName . $tdEnd;
		print $tdStart . $postText . $tdEnd;
		print $tdStart . $datePosted . $tdEnd;
		print $RowEnd;

		print $tableHeaders;

		for ($i = 0; $i < $cnt; ++$i) {	

			print $RowStart;
			print $tdStart . $replyData[$i]['member'] . $tdEnd;
			print $tdStart . $replyData[$i]['reply'] . $tdEnd;
			print $tdStart . $replyData[$i]['dateReplied'] . $tdEnd;
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
		print "<P align = center><A HREF = forumTest.php>Back to the Forum Topics</A></P>";
}
elseif ($numRows == 0) {
	print "This Forum is not yet available";
}
	}
	else {
		print "error displaying forum";
	}



}


?>



