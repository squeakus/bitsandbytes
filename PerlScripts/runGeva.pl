#!/usr/bin/perl -w

use strict;

my $result =0;
print "attempting to run GEVA!\n";

$result = `java -server -Xmx512m -cp Main.Run -jar /Users/jbyrne/GEVAExperimental/bin/GEVA.jar -properties_file /Users/jbyrne/GEVAExperimental/param/Parameters/Experiments/ShapeGrammar/PictureCopy.properties`;
print "Finished attempting to run GEVA!\n";

print("the result is ".$result."\n");
