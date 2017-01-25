#!/usr/bin/perl -w

#This script is to be used with the mutation operators that record the fitness
#of an individual before and after the operation.

package parseFitness;
use strict;

my $resultsDir = "/Users/jbyrne/MYGEVA/bin";

my $fileName;
my $popsize;

#read in the arguments passed in at the command line
if (@ARGV == 2) {
  $fileName = $ARGV[0];
  $popsize = $ARGV[1];
} else {
  print "please enter filename popsize\n";
  exit(0);
}
print("popsize ".$popsize."\n");

my $fullFilename= join("/",$resultsDir,$fileName);
my @fitnessArray;
my $counter =0;

my $line;

#build a list of all the directories
open(DAT, $fullFilename) or die "Can't open $fileName: $!";

while ($line = <DAT>)
{
  $counter++;
  #splitting the lines in the file into two variables  
  push(@fitnessArray,$line); 
}
close(DAT);

my $midpoint = $counter /2;
my @startArray;
my @midArray;
my @endArray;


for (my $count = 0; $count < $popsize; $count++)
{
	$startArray[$count] = $fitnessArray[$count]; 
}
my $tmpCount =0;
for (my $count = $midpoint; $count < ($midpoint + $popsize); $count++)
{
	$midArray[$tmpCount] = $fitnessArray[$count]; 
	$tmpCount++;
}
$tmpCount =0;
for (my $count =($counter - $popsize); $count < $counter; $count++)
{
	$endArray[$tmpCount] = $fitnessArray[$count]; 
	$tmpCount++;
}
print("Initial fitness:\n");
&parseArray(@startArray);
print("midpoint fitness:\n");
&parseArray(@midArray);
print("end fitness:\n");
&parseArray(@endArray);
sub parseArray
	{
	my $totalFitness =0;
	my $fitnessGain = 0;
	my $neutralCount =0;
	my $fitnessLoss =0;
	my $invalids =0;
	my $alreadyInvalids =0;
	my $gainCount =0;
	my $lossCount =0;
	
	foreach $line (@_)
	{
	 (my $field1,my $field2) = split ',', $line;
	
	 if($field2 ==1)
	  {
	  	$fitnessGain += $field1;
	  	$gainCount++;
	  }
	  elsif($field2==-1)
	  {
	  	$fitnessLoss += $field1;
	  	$lossCount++;
	  }
	  elsif($field2==0)
	  {
	  	$neutralCount++;
	  }
	  elsif($field2==-2)
	  {
	  	$invalids++;
	  }
	  elsif($field2==-3)
	  {
       $alreadyInvalids++;
	  }
	  $totalFitness += $field1;
	}
	print("The number of good mutations is: ".$gainCount."\n");
	print("The number of bad mutations is: ".$lossCount."\n");
	print("The number of neutral mutations is: ".$neutralCount."\n");
	print("The number of new invalids is: ".$invalids."\n");
	print("The number old invalids is: ".$alreadyInvalids."\n");
	print("the total fitness gain is: ".$fitnessGain."\n");
	print("the total fitness loss is: ".$fitnessLoss."\n");
	if($fitnessGain>0 && $gainCount>0)
	{
		print("the average fitness gain is: ".($fitnessGain/$gainCount)."\n");
	}
	else
	{
		print("the average fitness gain is: 0 \n");
	}
	print("the average fitness loss is: ".($fitnessLoss/$lossCount)."\n");
	print("the overall fitness change is: ".$totalFitness."\n");
	print("no of mutations: ".$counter."\n");
	}