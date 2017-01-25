#! /usr/bin/perl -w

use strict;

my $print = 0;
if(@ARGV > 3) {
    print "Usage: $0 dir expected print\n";
    exit(0);
}
if(@ARGV < 2){
print "Usage: $0 dir expected\n";
    exit(0);
}
my $dir = $ARGV[0];
if(!-e$dir) { die("No $dir\n"); }
my $expected = $ARGV[1];

my $out = "missing_".$dir.".txt";
if(@ARGV == 3) {
    $print = 1;
    open(OUT,">$out") || die("No write $out $!\n");
}

my $finished = 0;
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
	print "Missing: ".$diff." for ".$sub_dir."\n";
	if($print > 0) {
	    print OUT $basename ." " . $diff . "\n";
	}
    } else {
	$finished++;
	if($cnt > $expected) {
	    my $diff = $cnt - $expected;
	    print "More: ".$diff." for ".$sub_dir."\n";
	}
    }
}
print "Finished: ".$finished." for runs of ".$expected."\n";
if($print > 0) {
    close(OUT);
}
exit(0);
