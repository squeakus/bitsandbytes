#!/usr/bin/perl -w

use strict;

my $questionOutput = "/home/jonathan/Jonathan/programs/PHP/bridgesurvey/db/questions.csv";
my $answerOutput = "/home/jonathan/Jonathan/programs/PHP/bridgesurvey/db/answers.csv";
my $counter = 0;

open(QOP,">$questionOutput") || die("Cannot Open File"); 
open(AOP,">$answerOutput") || die("Cannot Open File"); 

for(my $i = 1; $i <= 100; $i++)
{
    #write that out to a file
    print(QOP $i.",");
    print(QOP "thumbs/back.".$i.".gif".",");
    print(QOP "thumbs/front.".$i.".gif"."\n");
    print(AOP $i.",");
    print(AOP "0".",");
    print(AOP "0".",");
    print(AOP "0"."\n");

}
close(QOP); 
close(AOP);
