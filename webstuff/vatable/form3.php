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
<div id="container">
  <div id="header">
    <h1>Justification</h1>
  </div>
  
  <div id="content">
    <p> Please give justification details: </p>
    <form method="post" action="form4.php">
      <textarea name="justification" rows="5" cols="40"></textarea>
      <br>
      <input type="submit" value="Submit">
  </form>
  </div>
  
  <div id="footer">
    Copyright: Vatable.com, 2015
  </div>
</div>

</body>
</html>