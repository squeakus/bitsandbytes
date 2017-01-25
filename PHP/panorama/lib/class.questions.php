<?php

class questions {

  var $ID;
  var $SelectType;
  var $Label;
  var $Ordering;
  
  function set_ID($ID) { $this->ID=$ID; }
  function get_ID() { return $this->ID; }

  function set_SelectType($SelectType) { $this->SelectType=$SelectType; }
  function get_SelectType() { return $this->SelectType; }

  function set_Label($Label) { $this->Label=$Label; }
  function get_Label() { return $this->Label; }

  function set_Ordering($Ordering) { $this->Ordering=$Ordering; }
  function get_Ordering() { return $this->Ordering; }

  function questions() {
  }

  function load($id) {
    $sql = sprintf("SELECT * FROM questions WHERE ID=%s", $id);
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    while($row = mysql_fetch_array($res, MYSQL_ASSOC)) {
      $this->ID=$row['ID'];
      $this->SelectType=$row['SelectType'];
      $this->Label=$row['Label'];
      $this->Ordering=$row['Ordering'];
    }
  }
  
  function insert() {
    $sql = sprintf("INSERT INTO questions (SelectType, Label, Ordering) VALUES (%s, %s, %s)", $this->SelectType, $this->Label, $this->Ordering);
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
    $sql = sprintf("UPDATE questions SET SelectType=%s, Label=%s, Ordering=%s WHERE ID=%s", $this->SelectType, $this->Label, $this->Ordering, $this->ID);
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete() {
    $sql = sprintf("DELETE FROM questions WHERE ID=%s", $this->ID);
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete_all() {
    $sql = "DELETE FROM questions";
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }
    
  function get_questions_id($start, $end) {
    $sth = "SELECT ID FROM questions ORDER BY Ordering LIMIT %s, %s";
    $sql = sprintf($sth, $start, $end);
    //echo $sql;
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    $ID = Array();
    while($row = mysql_fetch_array($res, MYSQL_NUM)) {
      $ID[] = $row[0];
    }
    return $ID;
  }

}

?>