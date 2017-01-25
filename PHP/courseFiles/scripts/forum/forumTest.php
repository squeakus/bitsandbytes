<?PHP

	$TableStart = "<TABLE height = 500 WIDTH = 600>";
	$TableEnd = "</TABLE>";
	$RowStart = "<TR >";
	$RowEnd = "</TR>";
	$tdStart = "<TD WIDTH = 200 height = 100 align = center valign = middle bgcolor =#EEEBEB>";
	$tdEnd = "</TD>";
	$tableHeaders = "<TR WIDTH = 200 height = 10 align = center valign = middle bgcolor =#00EBEB>";
	$tableHeaders = $tableHeaders . "<TD>Forum Sections</TD><TD>Posts</TD><TD>Replies</TD></TR>";

	$hrefStart = "<A HREF = pageThread.php?sID";

	$hrefEnd = "</A>";

	$secIDs[] = array();
	$tblPosts[] = array();
	$tblReply[] = array();
	$numPosts[] = array();
	$numReply[] = array();

	$tblPosts[] = "wpposts";
	$tblPosts[] = "wdposts";
	$tblPosts[] = "xlposts";
	$tblPosts[] = "vbposts";
	$tblPosts[] = "phposts";

	$tblReply[] = "wpreplies";
	$tblReply[] = "wdreplies";
	$tblReply[] = "xlreplies";
	$tblReply[] = "vbreplies";
	$tblReply[] = "phreplies";

	$user_name = "root";
	$password = "";
	$database = "dbforum";
	$server = "127.0.0.1";

	$db_handle = mysql_connect($server, $user_name, $password);
	$db_found = mysql_select_db($database, $db_handle);


	if ($db_found) {

		for ($i = 1; $i < 6; $i++) {

			//==================================================
			//	GET THE NUMBER OF POSTS IN EACH FORUM SECTION	
			//==================================================

			$SQL = "SELECT * FROM " . $tblPosts[$i];

			$result = mysql_query($SQL);

			if ($result) {

				$num_rows = mysql_num_rows($result);

				//==================================================		
				//	PUT THE NUMBER OF POSTS INTO AN ARRAY
				//==================================================
				$numPosts[$i] = $num_rows;

			}
			else {
				print "error";
			
			}
		}

			//=========================================================
			//	GET THE NUMBER OF REPLIES IN EACH FORUM SECTION
			//========================================================

			for ($i = 1; $i < 6; $i++) {
				$SQL = "SELECT * FROM " . $tblReply[$i];
				$result = mysql_query($SQL);

				if ($result) {

					$num_rows = mysql_num_rows($result);

					//==================================================		
					//	PUT THE NUMBER OF POSTS INTO AN ARRAY
					//==================================================
					$numReply[$i] = $num_rows;

				}
				else {
					print "error 2";
			
				}
			}

			//=======================================================
			//	GET THE FORUM MAIN TOPICS AND BUILD UP THE LINK
			//======================================================

			$SQL = "SELECT * FROM forumsections";
			$result = mysql_query($SQL);
			$loopCount = 1;

			while ($db_field = mysql_fetch_assoc($result)) {

				$secIDs[$loopCount] = $hrefStart . "=" . $db_field['sectionID'] . ">" . $db_field['sections'] . $hrefEnd;
				$loopCount++;
			}

			mysql_close($db_handle);

			//==================================================
			//	PRINT THE TABLE
			//==================================================
			print "<CENTER>";
			print $TableStart;
			print $tableHeaders;

			for ($i = 1; $i < 6; $i++) {

				print $RowStart;

					print $tdStart . $secIDs[$i] . $tdEnd;
					print $tdStart . $numPosts[$i] . $tdEnd;
					print $tdStart . $numReply[$i] . $tdEnd;

				print $RowEnd;
			}

			print $TableEnd;
			print "</CENTER>";
	}

?>


