<?php

function get_question_type($type) {
  switch ($type) {
  case 0:
    $ret = "radio";
    break;
  case 1:
    $ret = "checkbox";
    break;
  case 2:
    $ret = "textarea";
    break;
  }
  return $ret;
}

function printQuestion(&$nr, $questions) {
  global $cBox, $unansweredQ, $tArea;
  echo "\t<tr>\r\n".
    "\t\t<td class='litenrubrik' colspan='2' >";
  if(!empty($unansweredQ) && empty($_SESSION[addquotes($questions->{'ID'})])) {
    echo "<span class='highlight'>*</span>";
  }
  if(!empty($unansweredQ) && $questions->{'ID'} == $cBox) {
    if(sizeof($_SESSION[addquotes($cBox)]) != 3 || empty($_SESSION[addquotes($cBox)])) {
      echo "<span class='highlight'>* (3 alternativ eller enbart 'Nej, jag...') </span>";
    }
  }
  if($questions->{'ID'}==$tArea) {
    echo $questions->{'Label'} . "</td>\r\n".
					    "\t</tr>\r\n";
  } else {
    echo $nr.". ".$questions->{'Label'} . "</td>\r\n".
					    "\t</tr>\r\n";
    $nr++;
  }
}

function printAlternatives($type, $questions, $alternatives, $altIDs) {
  global $noCBoxAnswer, $cBox;
  foreach($altIDs as $AID) {
    $alternatives->load($AID);
    echo "\t<tr>\r\n";
    if($type == "checkbox") {
      if($alternatives->{'ID'} == $noCBoxAnswer && $questions->{'ID'} == $cBox) {
	echo "\t<tr>\r\n\t\t<td>&nbsp;</td>\r\n\t</tr>\r\n";
      }
      echo "\t\t<td class='text'><input type='".$type."' name='".$questions->{'ID'}."[]' value='".$alternatives->{'ID'}."'";
      if(!empty($_SESSION[addquotes($questions->{'ID'})]) && in_array($alternatives->{'ID'}, $_SESSION[addquotes($questions->{'ID'})]) ) {
	echo "checked";
      }
      echo "/></td>\r\n";
      echo "\t\t<td class='text'>".htmlentities($alternatives->{'Label'})."</td>\r\n";
    } else {
      if($type == "textarea") {
	echo "\t\t<td class='text'><".$type." cols='50' rows='10' name='".$questions->{'ID'}."' value='' ";
	echo "></".$type."></td>\r\n";
      } else {
	echo "\t\t<td class='text'><input type='".$type."' name='".$questions->{'ID'}."' value='".$alternatives->{'ID'}."'";
	if(!empty($_SESSION[addquotes($questions->{'ID'})]) && $_SESSION[addquotes($questions->{'ID'})] == $alternatives->{'ID'}) {
	  echo "checked";
	}
	echo "/></td>\r\n";
	echo "\t\t<td class='text'>".htmlentities($alternatives->{'Label'})."</td>\r\n"; 
      }
    }
    echo "\t</tr>\r\n";
  }
}

?>
<form method="POST" action="?page=submit&id=<?php echo($id + 1); ?>" >
<table border="0"> 
<colgroup>
<col width="20" span="1"/>
</colgroup>
<colgroup>
<col width="620" span="1"/>
</colgroup>
<?php
$alternatives = new alternatives();
$QID = $questions->get_questions_id($section[$id], $section[$id + 1] - $section[$id]);

if($id==0){
  echo "\t<tr>\r\n".
    "\t\t<td colspan='2' class='litenrubrik' >Undersökning av attityder till jämställdhet</td>\r\n" . 
    "\t</tr>\r\n";
  echo "\t<tr>\r\n".
    "\t\t<td colspan='2' ><p class='finstilt2' >Denna enkät består av " . (sizeof($questions->get_questions_id(0,1000)) - 1) . " frågor kring jämställdhet och tar cirka 15 minuter att slutföra. Du kan svara på frågorna i vilken ordning Du vill, men samtliga måste besvaras. Din medverkan är frivillig och anonym. Dina svar kommer inte att kunna härledas till Dig personligen.</p>" . 
    "<p class='finstilt2' >Tack för att Du tar Dig tid att medverka!</p></td>\r\n";
  echo "\t</tr>\r\n";
}
if($id < (sizeof($section) - 2)) { 
  echo "\t<tr>\r\n".
    "\t\t<td colspan='2' ><p class='finstilt' >Antalet frågor som visats eller besvarats ". sizeof($questions->get_questions_id(0,$section[$id + 1])) ." av " . (sizeof($questions->get_questions_id(0,1000)) - 1) . "</p></td>\r\n";
  echo "\t</tr>\r\n";
}

if($id==0){
  echo "\t<tr>\r\n".
    "\t\t<td class='litenrubrik' >Kön:";
  if(!empty($unansweredQ) && empty($_SESSION['Sex'])) {
    echo "<span class='highlight'>*</span>";
  }
  echo "</td>\r\n";
  echo "\t</tr>\r\n";
  echo "\t\t<td class='text'><input type='radio' name='Sex' value='Female'";
  if(!empty($_SESSION['Sex']) && $_SESSION['Sex'] == 'Female') {
    echo "checked";
  }
  echo "/></td>\r\n";
  echo "\t\t<td class='text'>Kvinna</td>\r\n";
  echo "\t</tr>\r\n";
  echo "\t<tr>\r\n";
  echo "\t\t<td class='text'><input type='radio' name='Sex' value='Male'";
  if(!empty($_SESSION['Sex']) && $_SESSION['Sex'] == 'Male') {
    echo "checked";
  }
  echo "/></td>\r\n";
  echo "\t\t<td class='text'>Man</td>\r\n";
  echo "\t</tr>\r\n";
}

$nr = $section[$id] + 1;
foreach($QID as $ID) {
  $questions->load($ID);
  $type = get_question_type($questions->{'SelectType'});
  $altIDs = $alternatives->get_alternatives_by_qid($questions->{'ID'}); 
  printQuestion($nr, $questions);
  printAlternatives($type, $questions, $alternatives, $altIDs);
}
echo "\t<tr>\r\n\t\t<td>&nbsp;</td>\r\n\t</tr>\r\n";
if($id == (sizeof($section) - 2)) { 
  echo "\t<tr>\r\n".
    "\t\t<td colspan='2' ><p class='finstilt' >Tack för din medverkan</p></td>\r\n" . 
    "\t</tr>\r\n";
}
?>
</table>
<?php if($id == (sizeof($section) - 2)) { ?>
 <input name="next" value="Registrera svar" type="submit" />
    <?php } else { ?>
 <input name="next" value="Nästa sida" type="submit" />
   <?php } ?>
</form>
