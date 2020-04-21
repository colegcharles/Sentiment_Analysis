<?php

?>


<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Machine Learning</title>
<!-- Latest compiled and minified CSS -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
<link rel="stylesheet" href='../css/home.css'>
<!-- jQuery library -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
<!-- Latest compiled JavaScript -->
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
<!--Geolocation -->
</head>
<body>
	<nav class="navbar navbar-default">
  <div class="container-fluid">
    <div class="navbar-header">
      <a class="navbar-brand" href="#">Sentiment Analysis Tool</a>
    </div>
    <ul class="nav navbar-nav">
      <li><a href="./home.php">Home</a></li>
    </ul>
  </div>
</nav>

<div>

	Welcome to the sentiment analysis tool. Every day the following data is updated to reflect how the following companys are perceived at the current date. We gather the data by querying twitter and applying a sentiment analysis algorithm.

</div>

<?php
$command = escapeshellcmd('python3 TestNB.py tacobell 100');
$output = shell_exec($command);
echo '<div>'.$output.'</div>';
?>
	
</body>
</html>