<?php
function printQuestion(&$nr, $questions) {
  global $tArea;
  echo "\t<tr>\r\n".
    "\t\t<td class='litenrubrik' colspan='3' >";
  if($questions->{'ID'}==$tArea) {
    echo $questions->{'Label'} . "</td>\r\n".
				   "\t</tr>\r\n";
  } else {
    echo $nr.". ".$questions->{'Label'} . "</td>\r\n".
					    "\t</tr>\r\n";
    $nr++;
  }
}

function printAlternatives($totNrAnswers, $questions, $alternatives, $altIDs, $answers) {
  global $cBox, $noCBoxAnswer;
  echo "\t<tr>\r\n";
  echo "\t\t<td class='finstilt'>K %</td>\r\n";
  echo "\t\t<td class='finstilt'>M %</td>\r\n";
  echo "\t\t<td>&nbsp;</td>\r\n";
  echo "\t</tr>\r\n";
  foreach($altIDs as $AID) {
    $alternatives->load($AID);
    $msg = "\t<tr>\r\n\t\t<td class='text_td' >%3.0f</td>\r\n\t\t<td class='text_td' >%3.0f</td>\r\n";
    $nrAnsw = $answers->get_nr_answers_id_sex($AID);
    if($totNrAnswers > 0) {
      $precF = ((int)$nrAnsw["Female"]/(int)$totNrAnswers)*100;
      $precM = ((int)$nrAnsw["Male"]/(int)$totNrAnswers)*100;
    } else {
      $precF = 0;
      $precM = 0;
    }

    if($alternatives->{'QID'}==$cBox) {
      if($alternatives->{'ID'}!=$noCBoxAnswer) {
	$precF = $precF/3;
	$precM = $precM/3;
      }
    }
    echo sprintf($msg,$precF, $precM);
    echo "\t\t<td class='text'>".htmlentities($alternatives->{'Label'})."</td>\r\n";
    echo "\t</tr>\r\n";
  }
}

function printComments() {
  $freetextanswers = new freetextanswers();
  $comments = $freetextanswers->show_answers();
  foreach($comments as $comment) {
    if($comment!="") {
      echo "\t<tr>\r\n";
      echo "\t\t<td class='text' >&nbsp;</td>\r\n\t\t<td class='text' >&nbsp;</td>\r\n";
      echo "\t\t<td class='text' >".$comment."</td>\r\n";
      echo "\t</tr>\r\n";
    }
  }
}

function printPages($section) {
    echo "\t<tr>\r\n";
    echo "\t\t<td class='litenrubrik' colspan='3'>Frågesida nr: ";
    for($i=0; $i<(sizeof($section) - 1); $i++) {
      echo "<a href='?page=questions&amp;bview=answers&amp;id=".$i."' >" . ($i+1) . "</a> ";
    }
    echo "</td>\r\n"; 
    echo "\t</tr>\r\n";
}
?>
<table border="0"> 
<colgroup>
<col width="20" span="2"/>
</colgroup>
<colgroup>
<col width="620" span="1"/>
</colgroup>
<?php
$alternatives = new alternatives();
$users = new users();
$answers = new answers();
$QID = $questions->get_questions_id($section[$id], $section[$id + 1] - $section[$id]);
$nrUsers = $users->get_nr_users();
if(is_array($nrUsers)) {
  $totUsers = array_sum($nrUsers);
} else {
  $totUsers = 0;
}

$msg = "\t<tr>\r\n\t\t<td class='text' colspan='3' ><p class='finstilt' >Antalet personer som svarat är %u, %01.1f%% kvinnor och %01.1f%% män.</p></td>\r\n";
if($totUsers == 0) {
  $fU = 0;
  $mU = 0;
} else {
  $fU = ($nrUsers["Female"]/$totUsers)*100;
  $mU = ($nrUsers["Male"]/$totUsers)*100;
}
echo sprintf($msg,$totUsers,$fU,$mU);
echo "\t</tr>\r\n";
printPages($section);

$nr = $section[$id] + 1;
foreach($QID as $ID) {
  $questions->load($ID);
  $altIDs = $alternatives->get_alternatives_by_qid($questions->{'ID'}); 
  printQuestion($nr, $questions);
  if($ID==$tArea) {
    printComments();
  } else {
    printAlternatives($totUsers, $questions, $alternatives, $altIDs, $answers);
  }
}
printPages($section);
?>
</table>
