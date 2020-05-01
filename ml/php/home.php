<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);
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

	Welcome to the sentiment analysis tool. Below, you can type in any company name or product and see the current sentiment analysis of the past 1000 tweets refering the that company. The sentiment analysis is calculated by a custom machine learning algorithm. once you search values will be calculated within 30 seconds.

</div>
<form action='' method='get'>

  <input name="searchWord" class="form-control" placeholder="Enter Company Name">
  <button class="btn btn-dark" type="submit">Search</button> 
</form>



<?php

if (isset($_GET["searchWord"])) {
  $command = escapeshellcmd('python3 testSent.py "'.$_GET["searchWord"].'"');
  $output = shell_exec($command);

  echo "<img src=\"./graphs/".$_GET["searchWord"]."PieGraphnew.png\" alt=\"smiley face\">";
  echo "<img src=\"./graphs/".$_GET["searchWord"]."ScatterPlotnew.png\" alt=\"smiley face\">";
  echo "<img src=\"./graphs/".$_GET["searchWord"]."LineGraphnew.png\" alt=\"smiley face\">";
  echo "<a href='./Data/".$_GET["searchWord"]."Datanew.csv' download>download the processed data</a>";
  
}
?>

	
</body>
</html>
