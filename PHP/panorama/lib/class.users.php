<?php

class users {

  var $ID;
  var $Sex;
  var $Start;
  var $End;
  
  function set_ID($ID) { $this->ID=$ID; }
  function get_ID() { return $this->ID; }

  function set_Sex($Sex) { $this->Sex=$Sex; }
  function get_Sex() { return $this->Sex; }

  function set_Start($Start) { $this->Start=$Start; }
  function get_Start() { return $this->Start; }

  function set_End($End) { $this->End=$End; }
  function get_End() { return $this->End; }

  function users() {
  }

  function load($id) {
    $sql = sprintf("SELECT * FROM users WHERE ID=%s", $id);
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    while($row = mysql_fetch_array($res, MYSQL_ASSOC)) {
      $this->ID=$row['ID'];
      $this->Sex=$row['Sex'];
      $this->Start=$row['Start'];
      $this->End=$row['End'];
    }
  }
  
  function insert() {
    $sql = sprintf("INSERT INTO users (Sex, Start, End) VALUES (%s, %s, %s)", $this->Sex, $this->Start, $this->End);
    //echo $sql;
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    $lastInsertId = "SELECT LAST_INSERT_ID()";
    $result = mysql_query($lastInsertId) or die(get_Class($this).':'."3 Could not save the entry to database!");
    while($row = mysql_fetch_array($result, MYSQL_NUM)) {
      $ID = $row[0];
    }
    mysql_free_result($result);
    return $ID;
  }

  function update() {
    $sql = sprintf("UPDATE users SET Sex='%s', Start='%s', End='%s' WHERE ID='%s'", $this->Sex, $this->Start, $this->End, $this->ID);
    echo $sql;
    mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete() {
    $sql = sprintf("DELETE FROM users WHERE ID=%s", $this->ID);
    mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete_all() {
    $sql = "DELETE FROM users";
    mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function get_nr_users() {
    $sql = "SELECT COUNT(*), Sex FROM users u GROUP BY Sex";
    //echo $sql;
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    while($row = mysql_fetch_array($res, MYSQL_NUM)) {
      $ID[$row[1]] = $row[0];
    }
    mysql_free_result($res);
    return $ID;
  }

}

?>