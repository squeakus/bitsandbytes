#!/usr/bin/perl -w

#This script is to be used with the mutation operators that record the fitness
#of an individual before and after the operation.

package parseFitness;
use strict;

my $resultsDir = "/Users/jbyrne/MYGEVA/bin";

my $fileName;
my $popsize;
my $mutOp = "nodal";

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

my $outputFile = join("","initial",$mutOp,"Change.dat");
open(OP,">>$outputFile") || die("Cannot Open File"); 
	foreach $line(@startArray)
	{ 
		print(OP $line);
	}
close(OP);

$outputFile = join("","middle",$mutOp,"Change.dat");
open(OP,">>$outputFile") || die("Cannot Open File"); 
	foreach $line(@midArray)
	{ 
		print(OP $line);
	}
close(OP);

$outputFile = join("","end",$mutOp,"Change.dat");
open(OP,">>$outputFile") || die("Cannot Open File"); 
	foreach $line(@endArray)
	{ 
		print(OP $line);
	}
close(OP);






