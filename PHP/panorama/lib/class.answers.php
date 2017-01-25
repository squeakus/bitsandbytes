<?php

class answers {

  var $ID;
  var $UID;
  var $QID;
  var $AID;
  
  function set_ID($ID) { $this->ID=$ID; }
  function get_ID() { return $this->ID; }

  function set_UID($UID) { $this->UID=$UID; }
  function get_UID() { return $this->UID; }

  function set_QID($QID) { $this->QID=$QID; }
  function get_QID() { return $this->QID; }

  function set_AID($AID) { $this->AID=$AID; }
  function get_AID() { return $this->AID; }

  function answers() {
  }

  function load($id) {
    $sql = sprintf("SELECT * FROM answers WHERE ID=%s", $id);
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    while($row = mysql_fetch_array($res, MYSQL_ASSOC)) {
      $this->ID=$row['ID'];
      $this->UID=$row['UID'];
      $this->QID=$row['QID'];
      $this->AID=$row['AID'];
    }
  }
  
  function insert() {
    $sql = sprintf("INSERT INTO answers (UID, QID, AID) VALUES (%s, %s, %s)", $this->UID, $this->QID, $this->AID);
    $res = mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
    $lastInsertId = "SELECT LAST_INSERT_ID()";
    $result = mysql_query($lastInsertId) or die(getClass($this).':'."3 Could not save the entry to database!");
    while($row = mysql_fetch_array($result, MYSQL_NUM)) {
      $ID = $row[0];
    }
    mysql_free_result($result);
    return $ID;
  }

  function update() {
    $sql = sprintf("UPDATE answers SET UID=%s, QID=%s, AID=%s WHERE ID=%s", $this->UID, $this->QID, $this->AID, $this->ID);
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete() {
    $sql = sprintf("DELETE FROM answers WHERE ID=%s", $this->ID);
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete_all() {
    $sql = "DELETE FROM answers";
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }
    
  function get_nr_answers_id($AID) {
    $sql = sprintf("SELECT COUNT(*) FROM answers WHERE AID=%s GROUP BY AID", $AID);
    //echo $sql;
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    while($row = mysql_fetch_array($res, MYSQL_NUM)) {
      $ID = $row[0];
    }
    mysql_free_result($res);
    return $ID;
  }

  function get_nr_answers_id_sex($AID) {
    $sql = sprintf('SELECT Sex, COUNT(answers.AID) FROM users INNER JOIN answers ON users.ID = answers.UID WHERE AID = %s GROUP BY users.Sex ORDER BY users.Sex ASC', $AID);
    //echo $sql;
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    while($row = mysql_fetch_array($res, MYSQL_NUM)) {
      $ID[$row[0]] = $row[1];
    }
    mysql_free_result($res);
    return $ID;
  }
}

?>