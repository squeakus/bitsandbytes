#!/usr/local/bin/perl -w

# Purpose
#       Reads in the file GUI and gives the numbering scheme
#       for the ouput file GUI.chart which is used to map out program
#       GUI.
#
#       use gui.pl -f GUI  > GUI.chart
#
use Getopt::Long;

# Set up the command line to accept a filename.
my $ret = GetOptions ("f|filename:s");
my $filename = $opt_f || die "Usage: $0 -f filename\n";

# Open the file.
open (INPUT, "$filename") || die "Could not open file $filename : $!\n";

# Start reading from the input file.
$col1 = -1;
$col2 = 0;
$col3 = 10;
while (<INPUT>)
{
   chop;

   # Write to the output filename.
   printf "%2d ", $col1;
   printf "%-31s  ", $_;
   printf "%2d  %4d", $col2, $col3;
   printf "\n";

   $col1++;
   $col2++;
   $col3 += 15;
}
printf "\n";
printf "Column 1 is for ControlText in *crtldsp.c \n";
printf "        and Color_flag in *crtlmse.c\n";
printf "Column 2 is for ControlDiv_y in *crtlmse.c\n";
printf "Column 3 is for textMove_y *crtldsp.c\n";
printf "\n";

# Close the files.
close (INPUT);
