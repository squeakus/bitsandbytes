#!/usr/bin/perl -w

use strict;

my $resultsDir;
my $fullFilename;
my $file;
my $folder;
my $GEVAfolder;
my @directoryArray;
my @fileArray;
my @numberArray;
my @resultsArray;
my $fitness;
my $actualfitness;
my $phenotype;
my $tmpString;
my $result;

if (@ARGV == 2) {
  ($folder,$GEVAfolder) = @ARGV;
} else {
  print "Usage: $0 resultsfolder GEVAfolder \n";
  exit(0);
}
$resultsDir = join("/",$GEVAfolder,"/ExperimentManager/",$folder);

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
        
	foreach my $line(@raw_data)
	{
	  if ($line =~ m/.*Phenotype:*/)
	    {
	      @numberArray = ($line =~ m/(\d+)/g);
	      if($line =~ m/Phenotype:(.*)/) 
	      {
	       $tmpString = $1
	      }
	      push(@resultsArray,$numberArray[1],$tmpString)
	    }
	} 
}
my $i;
chdir  "$GEVAfolder/GEVA/build/classes/"; 
for($i =0; $i< scalar(@resultsArray);$i++)
  {
   $fitness = $resultsArray[$i++];
   $phenotype = $resultsArray[$i];
   $actualfitness = `java FitnessEvaluation/MultiSquares/PictureCopy "$phenotype"`;
   #print "fitness: ".$fitness."\n";
   #print "phenotype: ".$phenotype."\n";
   if($actualfitness != $fitness){
    print "OH HOLY SHIT SOMETHING HAS GONE TERRIBLY TERRIBLY WRONG!?1?!?!?!!!!\n";
    print "phenotype: ".$phenotype."\n";
    print "supposed fitness: ".$fitness."\n";
    print "actual fitness: ".$actualfitness."\n";
  }
else
  {
    print "matched\n";
  }
  
  }
