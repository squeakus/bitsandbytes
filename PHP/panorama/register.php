<form method="post" action="?page=questions" name="frmLogin">
<table border="0"> 
  <tr>
   <td class="text" width="150">Användarnamn:</td>
   <td class="text"><input name="username" type="text" value="<?php if(!empty($_POST["username"])) { echo($_POST["username"]); } ?>" /></td>
  </tr>
  <tr>
   <td class="text" width="150">Lösenord:</td>
   <td class="text"><input name="password" type="password" value="" /></td>
  </tr>
  <tr>
   <td width="150">&nbsp;</td>
   <td class="text"><input class="text" name="btnLogin" type="submit" id="btnLogin" value="Logga in" /><input type="hidden" name="IsSent" value="true" /></td>
  </tr>
 </table>
</form>
