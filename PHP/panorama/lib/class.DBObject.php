<?php
//Identifier for all tables is ID
class DBObject
{
  var $ID = 0;
  var $table;
  var $fields = array();

  function __construct( $table, $fields )
  {
    $this->table = $table;
    foreach( $fields as $key )
      $this->fields[ $key ] = null;
  }

  function __get( $key )
  {
    return $this->fields[ $key ];
  }

  function __set( $key, $value )
  {
    if ( array_key_exists( $key, $this->fields ) )
    {
      $this->fields[ $key ] = $value;
      return true;
    }
    return false;
  }

  function load( $id )
  {
    $sth = "SELECT * FROM ".$this->table." WHERE ID=%s";
    $sql = vsprintf($sth, array( $id ));
    //echo $sql."\r\n";
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
    $row = mysql_fetch_array($res, MYSQL_ASSOC);
    $this->ID = $id;
    foreach( array_keys( $row ) as $key )
      $this->fields[ $key ] = $row[ $key ];
  }

  function insert()
  {
    $fields = "ID, ";
    $fields .= join( ", ", array_keys( $this->fields ) );

    $inspoints = array( "0" );
    foreach( array_keys( $this->fields ) as $field ) {
      $inspoints []= "%s";
    }
    $inspt = join( ", ", $inspoints );

    $sth = "INSERT INTO ".$this->table. 
      " ( $fields ) VALUES ( $inspt )";

    foreach( array_keys( $this->fields ) as $field ) {
      $values []= $this->fields[ $field ];
    }

    $sql = vsprintf($sth, $values );
    echo $sql."\r\n";
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());

    $sql = "SELECT last_insert_id()";
    $res = mysql_query($sql) or die(get_Class($this).':'."3 Could not save the entry to database!");
    while($row = mysql_fetch_array($res, MYSQL_NUM)) {
      $this->ID = $row[0];
    }
    return $this->ID;
  }

  function update()
  {
    $sets = array();
    $values = array();
    foreach( array_keys( $this->fields ) as $field )
    {
      $sets []= $field.'=%s';
      $values []= "'".$this->fields[ $field ]."'";
    }
    $set = join( ", ", $sets );
    $values []= $this->ID;

    $sth = 'UPDATE '.$this->table.' SET '.$set.
      ' WHERE ID=%s';

    $sql = vsprintf($sth, $values );
    //echo $sql."\r\n";
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete()
  {
    $sth = 'DELETE FROM '.$this->table.' WHERE ID=%s';
    $sql = vsprintf($sth, $this->ID);
    //    echo $sql."\r\n";
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
  }

  function delete_all()
  {
    $sql = 'DELETE FROM '.$this->table;
    //    echo $sql."\r\n";
    $res = mysql_query($sql) or die (get_Class($this).':'."I cannot query to the database because: " . mysql_error());
  }
}

class alternatives extends DBObject 
{
  function __construct()
  {
  parent::__construct( 'alternatives',
    array( 'QID', 'Label', 'Ordering' ) );
  }

  function get_alternatives_by_qid($qid) {
    $sth = "SELECT ID FROM ".$this->table." WHERE QID=%s";
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
    $sth = "SELECT ID FROM ".$this->table." WHERE QID=%s";
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

class answers extends DBObject 
{
  function __construct()
  {
  parent::__construct( 'answers',
    array( 'UID', 'QID', 'AID' ) );
  }
}

class freetextanswers extends DBObject 
{
  function __construct()
  {
  parent::__construct( 'freetextanswers',
    array( 'UID', 'QID', 'answer' ) );
  }
}

class questions extends DBObject 
{
  function __construct()
  {
  parent::__construct( 'questions',
    array( 'SelectType', 'Label', 'Ordering' ) );
  }

  function get_questions_id($start, $end) {
    $sth = "SELECT ID FROM ".$this->table." ORDER BY Ordering LIMIT %s, %s";
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

class users extends DBObject 
{

  function __construct()
  {
  parent::__construct( 'users',
    array( 'Sex', 'Start', 'End' ) );
  }

}

?>