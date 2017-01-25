<?PHP
$user_name = "root";
$password = "moot1254";
$database = "csisurvey";
$server = "127.0.0.1";	
$db_handle = mysql_connect($server, $user_name, $password);
$db_found = mysql_select_db($database, $db_handle);
$SQL = "SELECT SUM(FRONT),SUM(BACK),SUM(NOPREF) from answers;";
$result = mysql_query($SQL);
$db_field = mysql_fetch_array($result);
$backSum = $db_field["SUM(BACK)"];
$frontSum = $db_field["SUM(FRONT)"];
$noprefSum = $db_field["SUM(NOPREF)"];

$total = $backSum + $frontSum + $noprefSum;
$percentBack = (($backSum * 100) / $total);
$percentBack = round($percentBack,2);
$percentFront = (($frontSum * 100) / $total);
$percentFront = round($percentFront,2);
$percentNopref = (($noprefSum * 100) / $total);
$percentNopref = round($percentNopref,2);

$imgHeight = '10';
$imgWidthBack = $percentBack * 2;
$imgWidthFront = $percentFront * 2;
$imgWidthNopref = $percentNopref * 2;
$imgTagA = "<IMG SRC = 'red.jpg' Height = " . $imgHeight . " WIDTH = " . $imgWidthBack. ">";
$imgTagB = "<IMG SRC = 'red.jpg' Height = " . $imgHeight . " WIDTH = " . $imgWidthFront . ">";
$imgTagC = "<IMG SRC = 'red.jpg' Height = " . $imgHeight . " WIDTH = " . $imgWidthNopref . ">";
?>

<html>
<head>
<title>Summary of Survey Results</title>
<link rel="stylesheet" type="text/css" href="css/simple.css" />
</head>
<body>
   <h1> number of submissions:<?PHP print $total ?></h1>
   <p> The results so far are: </p>
   <h1> front pref:<?PHP print $frontSum ?></h1>
   <h1> back pref:<?PHP print $backSum ?></h1>
   <h1> no pref:<?PHP print $noprefSum ?></h1>
<center>
<?PHP
print "less constrained bridges:      " . $imgTagA . " " . $percentBack . "% " . $qA . "<BR>";
print "more constrained bridges:      " . $imgTagB . " " . $percentFront . "% " . $qB . "<BR>";
print "no preference:    " . $imgTagC . " " . $percentNopref . "% " . $qB . "<BR>";
?>
</center>
</body>
</html>
