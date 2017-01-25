function show_alert()
{
 add_two(5,6);
 alert("Hello! I am an awesome alert box!");
}

function add_two(a,b)
{
 number = a + b;
 alert('you have chosen'+ number);
}

function show_confirm()
{
var r=confirm("Press a button!");
if (r==true)
  {
  alert("You pressed OK!");
  }
else
  {
  alert("You pressed Cancel!");
  }
}

