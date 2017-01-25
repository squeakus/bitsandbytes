<?PHP
session_start();
if (!isset($_SESSION['curQuestion']))
{  
   $_SESSION['curQuestion']=1;
}

$qID = '';
$page = "";
$question = 'Question not set';
$user_name = "root";
$password = "";
$database = "csisurvey";
$server = "127.0.0.1";	
$nodalSum =0;
$structSum =0;


//setting up database
$db_handle = mysql_connect($server, $user_name, $password);
$db_found = mysql_select_db($database, $db_handle);
$result = mysql_query("SELECT * FROM questions");
if (!$result) 
{
   echo 'Could not run query: ' . mysql_error();
   exit;
}

//extracting data from DB
//$num_rows = mysql_num_rows($result);
$num_rows = 10;
$qNum = $_SESSION['curQuestion'];
$SQL = "SELECT * FROM questions WHERE questions.QID = '$qNum'";
$result = mysql_query($SQL);
$db_field = mysql_fetch_assoc($result);
$qID = $db_field['QID'];
$A = $db_field['A'];
$B = $db_field['B'];

$random =  rand(0,1);
if($random ==1)
  {
    $image1 = $A;
    $image2 = $B;
    $value1 = 'BACK';
    $value2 = 'FRONT';
  }
else
  {
    $image1 = $B;
    $image2 = $A;
    $value1 = 'FRONT';
    $value2 = 'BACK';
  }



// if an answer has been submitted then add it to the database;
if (!empty($_POST['q'])) 
{
  include('process.php');
}

// Control flow welcome -> survey -> thanks
if(!empty($_GET['page']))
{
  $page = $_GET['page'];
  if($page == "survey")
    {
      if($_SESSION['curQuestion'] < $num_rows) 
	{	 
	  $_SESSION['curQuestion'] +=1;
	  $action = "?page=survey";
	}
      else
	{
	  $_SESSION['curQuestion'] = 1;
	  $buttonName = "View-results";
	  $action = "?page=thanks";
	}
      include('survey.php');
    }
  else if($page == "thanks")
    {
      //getting values for graph
      $SQL = "SELECT SUM(FRONT),SUM(BACK),SUM(NOPREF) from answers;";
      $result = mysql_query($SQL);
      $db_field = mysql_fetch_array($result);
      $backSum = $db_field["SUM(BACK)"];
      $frontSum = $db_field["SUM(FRONT)"];
      $noprefSum = $db_field["SUM(NOPREF)"];
      include('thanks.php');
    }
} else {
  $_SESSION['curQuestion']=1;
  include('welcome.php');
}
mysql_close($db_handle);
?>
