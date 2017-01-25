<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
  <title> Vatable</title>
  <meta name="Description" content="VAT made easy" />
  <meta name="robots" content="all, index, follow" />
  <meta name="distribution" content="global" />
  <link rel="shortcut icon" href="/favicon.ico" />
  <link rel="stylesheet" href="style.css" type="text/css" />
</head>
<body> 

<?php
// define variables and set to empty values
$nameErr = $emailErr = $genderErr = $websiteErr = "";
$name = $email = $gender = $comment = $website = "";
$valid = true;

if ($_SERVER["REQUEST_METHOD"] == "POST") {
   if (empty($_POST["name"])) {
     $nameErr = "Name is required";
     $valid = false;
   } else {
     $name = test_input($_POST["name"]);
   }
   
   if (empty($_POST["email"])) {
     $emailErr = "Email is required";
     $valid = false;
   } else {
     $email = test_input($_POST["email"]);
     if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
       $emailErr = "Invalid email format";
       $valid = false;
     }
   }
     
   if (empty($_POST["website"])) {
     $website = "";
   } else {
     $website = test_input($_POST["website"]);
     // check if URL address syntax is valid (this regular expression also allows dashes in the URL)
     if (!preg_match("/\b(?:(?:https?|ftp):\/\/|www\.)[-a-z0-9+&@#\/%?=~_|!:,.;]*[-a-z0-9+&@#\/%=~_|]/i",$website)) {
       $websiteErr = "Invalid URL";
       $valid = false;
     }
   }
}

function test_input($data) {
   $data = trim($data);
   $data = stripslashes($data);
   $data = htmlspecialchars($data);
   return $data;
}
?>


<div id="container">
  <div id="header">
    <h1>User Details</h1>
  </div>
  
  <div id="content">
    <p><span class="error">* required field.</span></p>
    <form method="post" action="<?php if ($valid){ echo "form2.php";} else {echo htmlspecialchars($_SERVER["PHP_SELF"]);}  ;?>"> 
      Name: <input type="text" name="name" value="<?php echo $name;?>">
      <span class="error">* <?php echo $nameErr;?></span>
      <br><br>
      E-mail: <input type="text" name="email" value="<?php echo $email;?>">
      <span class="error">* <?php echo $emailErr;?></span>
      <br><br>
   Customer: <input type="text" name="website" value="<?php echo $website;?>">
      <span class="error"><?php echo $websiteErr;?></span>
      <br><br>
      <input type="submit" name="submit" value="Submit"> 
    </form>
  </div>
  <div id="footer">
    Copyright: Vatable.com, 2015
  </div>
    

</body>
</html>
