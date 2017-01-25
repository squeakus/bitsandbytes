#!/usr/bin/perl -w

package makeAOVArray;

use strict;

my $resultsDir = "/Users/jbyrne/MYGEVA/PRCConcatResults";
my $outputDir = "/Users/jbyrne/MYGEVA/PRCConcatResults";
my $outputFile = join("/",$outputDir,"collated.dat");
my $fullFilename;
my $file;
my @directoryArray;
my @fileArray;
my $range;
my $setSize;
my $result;

#build a list of all the directories
opendir(BIN, $resultsDir) or die "Can't open $resultsDir: $!";
while( defined ($file = readdir BIN) ) 
{
    if(($file =~ m/prc\d/)||
       ($file =~ m/prcRange/))
    {		
	$fullFilename = join("/",$resultsDir,$file);
    	push(@directoryArray,$fullFilename)
    }
}
closedir(BIN);

#build a list of all the files in those directories
foreach my $directoryName (@directoryArray) 
{
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
    
    if($fileName =~ m/.*Range*/)
    {
	$setSize = 1000;
    }
    else
    {
	$setSize = 100;
    }
    if($fileName =~ m/.*5\//)
    {
	$range = 5;
    }
    elsif($fileName =~ m/.*10\//)
    {
	$range = 10;
    }
    elsif($fileName =~ m/.*50\//)
    {
	$range = 50;
    }
    elsif($fileName =~ m/.*100\//)
    {
	$range = 100;
    }
    elsif($fileName =~ m/.*500\//)
    {
	$range = 500;
    }
    elsif($fileName =~ m/.*1000\//)
    {
	$range = 1000;
    }
    elsif($fileName =~ m/.*2000\//)
    {
	$range = 2000;
    }
    elsif($fileName =~ m/.*5000\//)
    {
	$range = 5000;
    }	 
    
    #write that out to a file
    open(OP,">>$outputFile") || die("Cannot Open File"); 
    print OP $counter." ".$setSize." ".$range." ".$result."\n";
    close(OP);	 
    $counter++;
}


