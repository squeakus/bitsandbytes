<html>
<head>
<title> Bridge design survey</title>
<link rel="stylesheet" type="text/css" href="css/simple.css" />
</head>


<body>
<h1> Question <?PHP print $qNum;?> of <?PHP print $num_rows;?></h1>
<FORM NAME ="form1" METHOD ="post" ACTION =<?PHP print $action;?>>

<p> click on the design you find most aesthetically pleasing or "No preference" if you have no preference</p>
<p>
<INPUT TYPE = 'Image'  Name ='q'  BORDER = "5" Src = <?PHP print $image1;?> value= <?PHP print $value1;?>>
<INPUT TYPE = 'Image'  Name ='q'  BORDER = "5" Src = <?PHP print $image2;?> value= <?PHP print $value2;?>>
</p>
<center><INPUT TYPE = "Image" Src ="thumbs/nopref.png" BORDER = "5" Name = 'q'  VALUE = 'NOPREF' ></center>
</FORM>
</body>
</html>










