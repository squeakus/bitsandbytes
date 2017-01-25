#!/usr/bin/perl -w

use strict;

my $result =0;
my $phenotype = "symX_( 100 ) [ [ symY_( -1 ) crcl ] ] [ [ [ [ [ [ LftRght_( -1 ) [ crcl ] ] ] ] ] ] ]";
$result = `java FitnessEvaluation/MultiSquares/PictureCopy "$phenotype"`;
print("the result is ".$result."\n");
