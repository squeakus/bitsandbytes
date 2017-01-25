<?php

function validateRegistration() {
  $msg = "";  //verify user
  foreach($_POST as $key => $value) {//get values from the POST array
    if($key == "username" || $key == "password") {
      if(!VerifyForm($value)) {
	$msg .= $key . " ";
      }
    }
  }
  return $msg;
}

function VerifyForm($string) {
  return ($string == "") ? 0 : 1;
}

function cleanString($value) {
  $value = trim($value);  //trim
  $value = htmlspecialchars($value);  //remove html
  if (get_magic_quotes_gpc()) {  // Stripslashes
    $value = stripslashes($value);
  }
  $value = "'" . mysql_real_escape_string($value) . "'";  // Quote if not integer
  return $value;
}

function addquotes($val){
  return "'" . $val . "'";
}

function removequotes($val){
  if(preg_match('/^(\')(\d+)(\')$/', $val, $matches)) {
    return $matches[2];
  }
}

?>