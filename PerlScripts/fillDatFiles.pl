#!/usr/bin/perl -w

package fillDatFiles;

use strict;

my $resultsDir = "/Users/jbyrne/MYGEVA/ExperimentManager";
my $fullFilename;
my $file;
my $folder;
my @directoryArray;
my @fileArray;
my $range;
my $setSize;
my $result;
my $count;
my $remove = 0;

if (@ARGV == 2) {
  ($folder, $count) = @ARGV;
} else {
  print "Usage: $0 folder count \n";
  exit(0);
}
$resultsDir = join("/",$resultsDir,$folder);

print("WARNING, this script might destroy your results, Do you want it to remove results?(y/n)");
my $usrInput = <STDIN>;
chomp($usrInput);
if($usrInput eq "y" ||
   $usrInput eq "yes")
   {
   	  $remove = 1;
   }


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
		if($file =~ m/.*dat/ )
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
	if(scalar(@raw_data) > $count)
	{
		print($fileName." contains more: ".scalar(@raw_data)."\n");
		my @new_data;
		if($remove)
		{			
			for(my $j =0; $j < $count; $j++)
			{
				$new_data[$j]= $raw_data[$j];
			}
		
			open(OP,">$fileName") || die("Cannot Open File");  
	   		print OP @new_data;
	    	close(OP);
		}	
	}
	
	if(scalar(@raw_data) < $count)
	{
	   print($fileName." contains less: ".scalar(@raw_data)."\n");	
	   if($remove)
	   {
		   my $missing = $count - scalar(@raw_data);
		   for(my $i =0; $i < $missing;$i++)
		   {
		      push(@raw_data,$raw_data[-1])
		   }
		  open(OP,">$fileName") || die("Cannot Open File");  
		  print OP @raw_data;
		  close(OP);	   
	   }
	} 
}