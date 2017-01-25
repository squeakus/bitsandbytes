#!/usr/bin/perl -w

use strict;

my $recip = 'jonathanbyrn@gmail.com'; 

# open a pipe to the mailx utility and feed it a subject 

#echo "This is the body."| mailx -s "mailx Test1" jonathanbyrn@gmail.com

use strict;
use warnings;

my $dir = "mutationTest";
my $to = 'jonathanbyrn@gmail.com';
my $subj = 'Results for experiment'.$dir;
my $body = "hey, it worked!!";
my @message;
push(@message,$body);
push(@message,$body);
push(@message,$body);

open (my $pipe, '|-', '/usr/bin/mailx', '-s', $subj, $to)
or die("can't open pipe to mailx: $!\n");
print($pipe @message);
close($pipe);
die "mailx exited with a non-zero status: $?\n" if $?;


