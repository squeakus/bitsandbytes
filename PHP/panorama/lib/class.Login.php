<?php

class Login {
  var $User;
  var $Password;
  var $Validated;

  function Login() {
    if(isset($_SESSION['logged'])){
      $this->Validated = true;
    }else {
      $this->Validated = false;
    }
  }

  function DoLogin($usr,$pass) {
    $sth = "SELECT * FROM admin WHERE login = %s AND password = %s";
    $sql = sprintf($sth, $usr, $pass);
    //echo $sql;
    $resultat = mysql_query($sql) or die('Did not work because: '.mysql_error());
    if(mysql_num_rows($resultat) > 0) {
      session_cache_expire(30);
      $_SESSION['logged'] = "TRUE";
    }
    mysql_free_result($resultat);
    $this->Login();
  }
		
  function DoLogout() {
    session_unset();
    session_destroy();
    $this->Validated = false;
    $this->User = '';
  }
}

?>