#!/usr/bin/perl -w

use strict;
my $GEVAfolder;
my $GEVAfitness;
my $fitness;
my $phenotype;
my $result;
my $runs = 0;
my $count =0;

if (@ARGV == 2) {
  ($GEVAfolder, $runs) = @ARGV;
} else {
  print "Usage: $0 GEVAFolder noOfRuns\n";
  exit(0);
}

while($count < $runs)
{
#run geva and store phenotype and fitness
#print"running GEVA\n";
$result = `java -server -Xmx512m -cp Main.Run -jar $GEVAfolder/bin/GEVA.jar -properties_file $GEVAfolder/param/Parameters/Experiments/ShapeGrammar/PictureCopy.properties`;

if ($result =~ m/.*Phenotype:*/)
{
      if($result =~ m/Fit:(\d+.\d+)/)
      {
       $GEVAfitness = $1;
      }               
      if($result =~ m/Phenotype:(.*)/)
      {
       $phenotype = $1;
      }
          
}

#running picturecopy and comparing fitness
chdir  "$GEVAfolder/GEVA/build/classes/";
$fitness = `java FitnessEvaluation/MultiSquares/PictureCopy "$phenotype"`;
chdir  "$GEVAfolder/bin";

if($GEVAfitness != $fitness)
  {
    print "OH HOLY SHIT SOMETHING HAS GONE TERRIBLY TERRIBLY WRONG!?1?!?!?!!!!\n";
    print "phenotype is ".$phenotype."\n";
    print "the supposed fitness is ".$GEVAfitness."\n";
    print "the actual fitness is ".$fitness;
  }
else
  {
    print "matched\n";
  }

$count++;
}
