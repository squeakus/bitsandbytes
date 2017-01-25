#!/usr/bin/perl -w

package fillDatFiles;

use strict;

my $resultsDir = "/Users/jbyrne/GEVAExperimental/ExperimentManager";
my $fullFilename;
my $file;
my $folder;
my @directoryArray;
my @fileArray;
my $range;
my $setSize;
my $result;
my $remove = 0;

if (@ARGV == 1) {
  ($folder) = @ARGV;
} else {
  print "Usage: $0 resultsFolder \n";
  exit(0);
}
$resultsDir = join("/",$resultsDir,$folder);

#build a list of all the directories
opendir(BIN, $resultsDir) or die "Can't open $resultsDir: $!";
while( defined ($file = readdir BIN) ) 
{
	$fullFilename = join("/",$resultsDir,$file);
    push(@directoryArray,$fullFilename)
}
closedir(BIN);

#build a list of all the files in those directories
foreach my $directoryName (@directoryArray) {
opendir(BIN, $directoryName) or die "Can't open $directoryName: $!";
	while( defined ($file = readdir BIN) ) 
	{
		if($file =~ m/.*out/ )
		{		
			$fullFilename = join("/",$directoryName,$file);	
	   	 	push(@fileArray,$fullFilename)
		}
	}
	closedir(BIN);
}	

#open each file and check if the have all the elements
foreach my $fileName (@fileArray) 
{
  	#open the file to get the data
	open(DAT,$fileName) || die("Cannot Open File"); 
	my @raw_data=<DAT>;
	close(DAT); 
        
	print "\n".$fileName."\n";
	foreach my $line(@raw_data)
	{
	  if ($line =~ m/.*Phenotype:*/)
	    {
	      print $line
	    }
	} 
}
