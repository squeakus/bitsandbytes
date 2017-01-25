#!/usr/bin/perl -w

package makeAOVArray;

use strict;
use Switch;

my $resultsDir = "/Users/jbyrne/MYGEVA/ExperimentManager/galapagosResults/MutationTest";
my $experiment = "wm";
my $outputFile = "/Users/jbyrne/MYGEVA/ExperimentManager/galapagosResults/".$experiment."Collated.dat";
my $fullFilename;
my $file;
my @directoryArray;
my @fileArray;
my $mutOp;
my $mutRate;
my $crossoverRate;
my $result;

#build a list of all the directories
opendir(BIN, $resultsDir) or die "Can't open $resultsDir: $!";
while( defined ($file = readdir BIN) ) 
{
	if(($file =~ m/$experiment/)&&!($file =~ m/eps/))
	{		
		$fullFilename = join("/",$resultsDir,$file);
    	push(@directoryArray,$fullFilename)
	}
}
closedir(BIN);

#build a list of all the files in those directories
foreach my $directoryName (@directoryArray) {
opendir(BIN, $directoryName) or die "Can't open $directoryName: $!";
	while( defined ($file = readdir BIN) ) 
	{
		if($file =~ m/.*dat/ )
		{		
			$fullFilename = join("/",$directoryName,$file);	
	   	 	push(@fileArray,$fullFilename)
		}
	}
	closedir(BIN);
}	

my $counter =0;

#clear output file
open(OP,">$outputFile") || die("Cannot Open File"); 
close(OP);

#open each file and read out the best result
foreach my $fileName (@fileArray) 
{	
   $mutOp = "moo";
   $mutRate = "";
   $crossoverRate = "";
	
	#open the file to get the data
	open(DAT,$fileName) || die("Cannot Open File"); 
	my @raw_data=<DAT>;
	close(DAT); 
		
	#extract the best result
	my @elementList = split(/\s/,$raw_data[-1]);
	$result = $elementList[0];
	
	 switch($fileName) {
     case /varmut/     { $mutOp = "VarMut"; next }
     case /intflip/    { $mutOp = "IntFlip"; next }
     case /struct/     { $mutOp = "Struct"; next }
     case /node/       { $mutOp = "Node"; next }
     case /xo_0.9/     { $crossoverRate = "0.9"; next }
     case /xo_0.0/     { $crossoverRate = "0.0"; next }
     case /m_0.0/      { $mutRate = "0.0"; next }
     case /m_0.01/     { $mutRate = "0.01"; next }
     case /m_0.1/      { $mutRate = "0.1"; next }
     case /m_0.15/     { $mutRate = "0.15"; next }
     case /m_0.2/      { $mutRate = "0.2"; next }
	 }
	 if(!($mutOp =~ "moo"))
	 {
	 	#write that out to a file
	 	open(OP,">>$outputFile") || die("Cannot Open File"); 
		print OP $counter." ".$mutOp." ".$crossoverRate." ".$mutRate." ".$result."\n";
		close(OP);
		$counter++;
	 }
	      
}


