#!/usr/bin/perl -w

use strict;

my $file1;
my $file2;

if (@ARGV == 2) {
  ($file1,$file2) = @ARGV;
} else {
  print "Usage: $0 file1 file2 \n";
  exit(0);
}


open(DAT,$file1) || die("Cannot Open File"); 
my @fileData1=<DAT>;
close(DAT);

open(DAT,$file2) || die("Cannot Open File"); 
my @fileData2=<DAT>;
close(DAT);

my $diffcount =0;

for(my $i =0; $i < scalar(@fileData1); $i++)
  {
    if($fileData1[$i] != $fileData2[$i])
      {
	$diffcount++;
      }
  }
print "The amount of differences is: ".$diffcount."\n";			



