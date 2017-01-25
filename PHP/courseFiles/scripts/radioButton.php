<html>
<head>
<title>Radio Buttons</title>
</head>

<?PHP

$male_status = 'unchecked';
$female_status = 'unchecked';


if (isset($_POST['Submit1'])) {

	$selected_radio = $_POST['gender'];
	
		if ($selected_radio == 'male') {
			$male_status = 'checked';

		}
		else if ($selected_radio == 'female') {
			$female_status = 'checked';
		}
}

?>

<body>

<FORM NAME ="form1" METHOD ="POST" ACTION ="radioButton.php">

<INPUT TYPE = 'Radio' Name ='gender'  value= 'male' <?PHP print $male_status; ?>>Male

<INPUT TYPE = 'Radio' Name ='gender'  value= 'female' <?PHP print $female_status; ?>>Female
<P>
<INPUT TYPE = "Submit" Name = "Submit1"  VALUE = "Select a Radio Button">
</FORM>

</body>
</html>


