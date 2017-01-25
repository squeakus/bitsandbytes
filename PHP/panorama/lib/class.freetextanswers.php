<?php

class freetextanswers {

  var $ID;
  var $UID;
  var $QID;
  var $answer;
  
  function set_ID($ID) { $this->ID=$ID; }
  function get_ID() { return $this->ID; }

  function set_UID($UID) { $this->UID=$UID; }
  function get_UID() { return $this->UID; }

  function set_QID($QID) { $this->QID=$QID; }
  function get_QID() { return $this->QID; }

  function set_answer($answer) { $this->answer=$answer; }
  function get_answer() { return $this->answer; }

  function freetextanswers() {
  }

  function load($id) {
    $sql = sprintf("SELECT * FROM freetextanswers WHERE ID=%s", $id);
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    while($row = mysql_fetch_array($res, MYSQL_ASSOC)) {
      $this->ID=$row['ID'];
      $this->UID=$row['UID'];
      $this->QID=$row['QID'];
      $this->answer=$row['answer'];
    }
  }
  
  function insert() {
    $sql = sprintf("INSERT INTO freetextanswers (UID, QID, answer) VALUES (%s, %s, %s)", $this->UID, $this->QID, $this->answer);
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
    $sql = sprintf("UPDATE freetextanswers SET UID=%s, QID=%s, answer=%s WHERE ID=%s", $this->UID, $this->QID, $this->answer, $this->ID);
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete() {
    $sql = sprintf("DELETE FROM freetextanswers WHERE ID=%s", $this->ID);
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete_all() {
    $sql = "DELETE FROM freetextanswers";
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }
    
  function show_answers() {
    $sql = sprintf("SELECT answer FROM freetextanswers");
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    while($row = mysql_fetch_array($res, MYSQL_ASSOC)) {
      $answers[] = $row['answer'];
    }
    return $answers;
  }

}

?>