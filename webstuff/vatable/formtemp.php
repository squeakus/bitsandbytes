<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
  <title> Vatable</title>
  <meta name="Description" content="VAT made easy" />
  <meta name="robots" content="all, index, follow" />
  <meta name="distribution" content="global" />
  <link rel="shortcut icon" href="/favicon.ico" />
  <link rel="stylesheet" href="style.css" type="text/css" />
</head>

<?php 
session_start();
$_SESSION['membership'] = $_POST['membership'];
$_SESSION['terms'] = $_POST['terms'];
?>

<body> 
  <div id="header">
    <h1>Form 3</h1>
  </div>
  
  <div id="content">
    <p> Hi </p>
  </div>
  
    <?php
    echo "<h2>Your Membership:</h2>";
    echo $_SESSION['membership'];
    echo "<br>";
    echo $_SESSION['terms'];
    ?>


</body>
</html>
