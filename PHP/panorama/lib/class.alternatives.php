<?php

class alternatives {

  var $ID;
  var $Label;
  var $QID;
  var $Ordering;
  
  function set_ID($ID) { $this->ID=$ID; }
  function get_ID() { return $this->ID; }

  function set_Label($Label) { $this->Label=$Label; }
  function get_Label() { return $this->Label; }

  function set_QID($QID) { $this->QID=$QID; }
  function get_QID() { return $this->QID; }

  function set_Ordering($Ordering) { $this->Ordering=$Ordering; }
  function get_Ordering() { return $this->Ordering; }

  function alternatives() {
  }

  function load($id) {
    $sql = sprintf("SELECT * FROM alternatives WHERE ID=%s", $id);
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    while($row = mysql_fetch_array($res, MYSQL_ASSOC)) {
      $this->ID=$row['ID'];
      $this->Label=$row['Label'];
      $this->QID=$row['QID'];
      $this->Ordering=$row['Ordering'];
    }
  }
  
  function insert() {
    $sql = sprintf("INSERT INTO alternatives (Label, QID, Ordering) VALUES (%s, %s, %s)", $this->Label, $this->QID, $this->Ordering);
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
    $sql = sprintf("UPDATE alternatives SET Label=%s, QID=%s, Ordering=%s WHERE ID=%s", $this->Label, $this->QID, $this->Ordering, $this->ID);
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete() {
    $sql = sprintf("DELETE FROM alternatives WHERE ID=%s", $this->ID);
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete_all() {
    $sql = "DELETE FROM alternatives";
    mysql_query($sql) or die (getClass($this).':'."I cannot query to the database because: " . mysql_error());
  }
    
  function get_alternatives_by_qid($qid) {
    $sth = "SELECT ID FROM alternatives WHERE QID=%s";
    $sql = vsprintf($sth, array( $qid ));
    //    echo $sql."\r\n";
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    $ID = Array();
    while($row = mysql_fetch_array($res, MYSQL_NUM)) {
      $ID[] = $row[0];
    }
    return $ID;
  }
  function get_alternative_by_qid($qid) {
    $sth = "SELECT ID FROM alternatives WHERE QID=%s";
    $sql = vsprintf($sth, array( $qid ));
    //    echo $sql."\r\n";
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    $ID;
    while($row = mysql_fetch_array($res, MYSQL_NUM)) {
      $ID = $row[0];
    }
    return $ID;
  }

}

?>