#! /usr/bin/perl -w

use strict;
my @message;

my $print = 0;

if(@ARGV != 2){
print "Usage: $0 dir expected\n";
    exit(0);
}

my $dir = $ARGV[0];
if(!-e$dir) { die("No $dir\n"); }
my $expected = $ARGV[1];

my $finished = 0;
my $lines;

while(<$dir/*/>) {
    my $sub_dir = $_;
    my $cnt = 0;
    
    my $basename = "";
 if($sub_dir =~ m/^.*\/(.*)\//) {
  $basename = $1;
 }
    while(<$sub_dir/*dat>) {
	$cnt++;
    }
    if($cnt < $expected) {
	my $diff = $expected - $cnt;
	push(@message,"Missing: ".$diff." for ".$sub_dir."\n");
    } else {
	$finished++;
	if($cnt > $expected) {
	    my $diff = $cnt - $expected;
	    push(@message,"More: ".$diff." for ".$sub_dir."\n");
	}
    }
}
push(@message,"Finished: ".$finished." for runs of ".$expected."\n");

my $to = 'jonathanbyrn@gmail.com';
my $subj = 'Results for experiment: '.$dir;

open (my $pipe, '|-', '/usr/bin/mailx', '-s', $subj, $to)
or die("can't open pipe to mailx: $!\n");
print($pipe @message);
close($pipe);
die "mailx exited with a non-zero status: $?\n" if $?;

exit(0);
