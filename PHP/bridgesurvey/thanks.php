<?PHP
$imgHeight = '10';
$total = '';
$percentBack = '0';
$percentFront = '0';
$percentNopref = '0';

//calculating percentages
$total = $backSum + $frontSum + $noprefSum;
$percentBack = (($backSum * 100) / $total);
$percentBack = round($percentBack,2);
$percentFront = (($frontSum * 100) / $total);
$percentFront = round($percentFront,2);
$percentNopref = (($noprefSum * 100) / $total);
$percentNopref = round($percentNopref,2);

$imgWidthBack = $percentBack * 2;
$imgWidthFront = $percentFront * 2;
$imgWidthNopref = $percentNopref * 2;

//creating graph
$imgTagA = "<IMG SRC = 'red.jpg' Height = " . $imgHeight . " WIDTH = " . $imgWidthBack. ">";
$imgTagB = "<IMG SRC = 'red.jpg' Height = " . $imgHeight . " WIDTH = " . $imgWidthFront . ">";
$imgTagC = "<IMG SRC = 'red.jpg' Height = " . $imgHeight . " WIDTH = " . $imgWidthNopref . ">";
?>

<html>
<head>
<title>Survey Complete</title>
<link rel="stylesheet" type="text/css" href="css/simple.css" />
</head>
<body>
   <h1> Thank you for taking part in this survey </h1>
   <p> The purpose of this survey was to see if adding constraints, such as limits on material or structural integrity, affect the quality of the generated designs. </p>
   <p> The results so far are: </p>

<center>
<?PHP
print "less constrained bridges:      " . $imgTagA . " " . $percentBack . "% " . $qA . "<BR>";
print "more constrained bridges:      " . $imgTagB . " " . $percentFront . "% " . $qB . "<BR>";
print "no preference:    " . $imgTagC . " " . $percentNopref . "% " . $qB . "<BR>";
?>
</center>
</body>
</html>
