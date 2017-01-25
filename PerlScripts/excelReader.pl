#! /usr/bin/perl -w

use Spreadsheet::ParseExcel::Simple;
use strict;

my $xls = Spreadsheet::ParseExcel::Simple->read("./test1.xlsx");
foreach my $sheet($xls->sheets) 
{
while ($sheet->has_data) 
{
my @data = $sheet->next_row;
print "@data\n";
}
}
