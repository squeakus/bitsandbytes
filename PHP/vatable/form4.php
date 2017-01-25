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
$_SESSION['justification'] = $_POST['justification'];
$var = "mooooooo";
echo "moooooo";
$result = exec("/usr/bin/python writetofile.py .$var");
echo $result;
$result2 = exec("whoami");
echo $result2;
?>

<body> 
  <div id="header">
    <h1>Finished!</h1>
  </div>
  

  <div id="content">
    <p> Well done! you have completed the form, here are your details</p>
   <?php
   echo "<h2>Name:</h2>";
   echo $_SESSION['name'];
   echo "<h2>Email:</h2>";
   echo $_SESSION['email'];
   echo "<h2>Terms:</h2>";
   echo $_SESSION['terms'];
   echo "<h2>Justification:</h2>";
   echo $_SESSION['justification'];
   ?>

    <p> Click the download button to receive a report</p>
    <button>
    <a href="moo.txt" download="moo.txt">Download</a>
  </button>

  </div>



</body>
</html>
