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
$_SESSION['name'] = $_POST['name'];
$_SESSION['email'] = $_POST['email'];
?>

<body> 
<div id="container">
  <div id="header">
    <h1>Terms and conditions</h1>
  </div>
  
  <div id="content">
    <p> Hi! </p>
    <form method="post" action="form3.php">
      <input type="radio" name="membership" value="Free">Free
      <br>
      <input type="radio" name="membership" value="Normal">Normal
      <br>
      <input type="radio" name="membership" value="Deluxe">Deluxe
      <br>
      <input type="checkbox" name="terms">I accept the terms and conditions
      <br>
      <input type="submit" value="Go To Justification">
    </form>
  </div>
  <div id="footer">
    Copyright: Vatable.com, 2015
  </div>
</div>

</body>
</html>
