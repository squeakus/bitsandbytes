<?php
// This is an example opendb.php
$conn = mysql_connect("localhost", "fablenu_panorama", "panorama") or die ('Error connecting to mysql. ' . mysql_error());
mysql_select_db("fablenu_panorama");
?>