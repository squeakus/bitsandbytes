#!/usr/bin/perl -w

package calcFitness;
use strict;

my $fileName;
if (@ARGV == 1) {
  ($fileName) = @ARGV;
} else {
  print "please enter filename\n";
  exit(0);
}


my $fitnessDir = "/Users/jbyrne/MYGEVA/ExperimentManager";
my $fullFilename= join("/",$fitnessDir,$fileName);
my @fitnessArray;
my $counter =0;
my $totalFitness =0;
my $invalids =0;

#build a list of all the directories
open(DAT, $fullFilename) or die "Can't open $fileName: $!";
@fitnessArray=<DAT>;
close(DAT);

foreach(@fitnessArray)
{
	$counter++;
	if($_ == -1)
	{
		$invalids++;
	}
	else{
	$totalFitness += $_;
	}
}
my $averageFitness = $totalFitness/$counter;

print("totalFitness:".$totalFitness."\n");
print("averageFitness:".$averageFitness."\n");
print("count:".$counter."\n");
print("invalids:".$invalids."\n");




