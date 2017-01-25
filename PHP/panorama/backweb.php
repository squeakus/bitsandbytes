<?php
session_start();

include('lib/class.alternatives.php');
include('lib/class.answers.php');
include('lib/class.users.php');
include('lib/class.freetextanswers.php');
include('lib/class.questions.php');
include('lib/functions.php');
$dbh = mysql_connect('localhost', 'fabelnu_panorama', 'panorama') or die ('Error connecting to mysql. ' . mysql_error());
mysql_select_db('fabelnu_panorama_development');
//$dbh = mysql_connect('localhost', 'fabelnu_enresa', 'enresa') or die ('Error connecting to mysql. ' . mysql_error());
//mysql_select_db('fabelnu_panorama');

$users = new users();
$answers = new answers();
$questions = new questions();
$type = Array("radio", "checkbox");
$section = Array(0,13,21,29,31,32);
$tArea = 32; //questino with textarea
$cBox = 30; //question with checkBox
$noCBoxAnswer = 156; //not ansewring the cBox question

// Check login
if(!empty($_GET['page'])){
  $page = $_GET['page'];
  if($page == "questions"){
    include('lib/class.Login.php');
    $lh = new Login();
    if(!empty($_POST['IsSent']) && $_POST["username"] == "backweb"){ //HCK for username
      $lh->DoLogin(cleanString($_POST["username"]),cleanString($_POST["password"]));
    }
    if(!empty($_GET['action'])){
      if($_GET['action'] == "logout"){
	$lh->DoLogout();
	header('Location: index.php');
	exit;
      }
    }
  }
}

?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>Panorama enkät</title>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
<link rel="stylesheet" type="text/css" href="css/mall.css" />
</head>
<body>
	<table width="100%"> 
		<tr><td valign="top" align="center">
	<table bgcolor="#999966" width="520" border="0" cellpadding="0" cellspacing="0">
		<tr>
			<td class="gg-h" valign="top" align="left"><img src="images/top_left.gif" width="12" height="24" alt="top_left"/></td>	
			<td colspan="3" height="10" align="right">
				<table width="100%" cellspacing="0" cellpadding="0" border="0">
					<tr>
						<td align="left" valign="top" width="10"><img src="images/top_left2.gif" width="10" height="3" alt="top_left2"/></td>
						<td><p class="finstilt" align="right">Panorama enkät</p></td>
						<td align="right" valign="top" width="10"><img src="images/top_right2.gif" width="10" height="3" alt="top_right2"/></td>
					</tr>
				</table>
			</td>
			<td class="gg-h" valign="top" align="right"><img src="images/top_right.gif" width="12" height="24" alt="top_right"/></td>
		</tr>
		<tr>
			<td class="gg-h"></td>	
			<td colspan="3" class="g" align="right" valign="top"><img src="images/header.gif" width="520" height="102" alt="header"/></td>
			<td class="gg-h"></td>
		</tr>
		<tr>
			<td class="gg-h"></td>
			<td colspan="3" height="10" class="gg"></td>
			<td class="gg-h"></td>
		</tr>
		<tr>
			<td class="gg-h"></td>	
			<td colspan="3" class="g" align="right" valign="top">|<a href="?page=questions&amp;action=logout" > logga ut</a> </td>
			<td class="gg-h"></td>
		</tr>
		<tr>
			<td class="gg-h"></td>
			<td colspan="3" height="10" class="gg"></td>
			<td class="gg-h"></td>
		</tr>
		<tr>
			<td class="gg-h"></td>
			<td colspan="3" class="g">

<?php

if($page == "questions") { 
  if(!empty($_POST["IsSent"])) {
    $msg = validateRegistration();
    if($msg == "") {
      if($lh->Validated) {
	$id = 0;
	include('answers.php');
      } else {
	echo "<p class='text' >Lösen eller användare fel</p>\n\r";
	include('register.php');
      }
    } else {
      echo "<p class='text' >" .$msg . "</p>\n\r";
      include('register.php');
    }
  } else {
    if($lh->Validated) {
      if(!empty($_GET['bview'])){
	$bview = $_GET['bview'];
	if($bview == "answers") {
	  $id = $_GET['id'];
	  include('answers.php');
	}
      }
    }
  }
} else {
  include('register.php');
}

mysql_close($dbh);

?>
                </td>
		</tr>
				
		<tr>
			<td class="gg-h" valign="bottom"><img src="images/bottom_left.gif" width="12" height="12" alt="bottom_left"/></td>
			<td colspan="3" height="10" align="center"><p class="finstilt">&#xA9; 2006 Fabel - Kastellgatan 18 - 413 07 GBG- 031-761 03 32 - <a href="mailto:info@fabel.se">info@fabel.se</a></p></td>
			<td class="gg-h" valign="bottom" align="right"><img src="images/bottom_right.gif" width="12" height="12" alt="bottom_right"/></td>
		</tr>

	</table>
	
			</td >
		</tr>
	</table>
</body>
</html>
