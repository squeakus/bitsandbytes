#!/usr/bin/perl -w

package makeAOVArray;

use strict;

my $resultsDir = "/Users/jonathanbyrne/results/maxAdjusted";
my $outputDir = "/Users/jonathanbyrne/results";
my $outputFile = join("/",$outputDir,"collated.dat");
my $fullFilename;
my $file;
my @directoryArray;
my @fileArray;
my $mutOp = 0;
my $result;

#build a list of all the directories
opendir(BIN, $resultsDir) or die "Can't open $resultsDir: $!";
while( defined ($file = readdir BIN) ) 
{
	if($file =~ m/.*depth8/)
	{		
	$fullFilename = join("/",$resultsDir,$file);
    	print("files being processed: ".$fullFilename."\n");
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

#open each file and read out the best result
foreach my $fileName (@fileArray) 
{		
    #open the file to get the data
    open(DAT,$fileName) || die("Cannot Open File"); 
    my @raw_data=<DAT>;
    close(DAT); 
	
    #extract the best result
    my @elementList = split(/\s/,$raw_data[-1]);
    $result = $elementList[0];
    if($fileName =~ m/.*intflip/)
    {
	$mutOp = "intflip";
    }
    elsif($fileName =~ m/.*node/)
    {
	$mutOp = "node";
    }
    elsif($fileName =~ m/.*struct/)
    {
	$mutOp = "struct";
    }
    elsif($fileName =~ m/.*context/)
    {
	$mutOp = "subtree";
    }
    #write that out to a file
    open(OP,">>$outputFile") || die("Cannot Open File"); 
    print OP $counter." ".$mutOp." ".$result."\n";
    close(OP);	 
    $counter++;
}


