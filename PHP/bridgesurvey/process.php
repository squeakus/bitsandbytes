<?PHP
$selected_answer = $_POST['q'];	    

if ($db_found) {
    $_SESSION['hasVoted'] = '1';
    $SQL = "UPDATE answers SET $selected_answer = $selected_answer + 1 where ID = $qNum";
    $result = mysql_query($SQL);
    if (!$result) 
      {
	echo 'Could not run query: ' . mysql_error();
	exit;
      }
  }
else {
    print "database error";
  }
?>

