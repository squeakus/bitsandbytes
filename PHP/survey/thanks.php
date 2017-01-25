<?PHP
$imgHeight = '10';
$totalP = '';
$percentBack = '0';
$percentFront = '0';
$percentNopref = '0';

//calculating percentages
$totalP = $backSum + $frontSum + $noprefSum;
$percentBack = (($backSum * 100) / $totalP);
$percentBack = floor($percentBack);
$percentFront = (($frontSum * 100) / $totalP);
$percentFront = floor($percentFront);
$percentNopref = (($noprefSum * 100) / $totalP);
$percentNopref = floor($percentNopref);

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
<link rel="stylesheet" type="text/css" href="css/mall.css" />
</head>
<body>
   <p> Thank you for taking part in this survey </p>
   <p> The results so far are: </p>

<center>
<?PHP
print "constrained:      " . $imgTagA . " " . $percentBack . "% " . $qA . "<BR>";
print "unconstrained:    " . $imgTagB . " " . $percentFront . "% " . $qB . "<BR>";
print "no preference:    " . $imgTagC . " " . $percentNopref . "% " . $qB . "<BR>";
?>
</center>
</body>
</html>