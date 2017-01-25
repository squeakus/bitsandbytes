#! /usr/bin/perl -w

use Text::Iconv;
my $converter = Text::Iconv -> new ("utf-8", "windows-1251");
my $docsFolder = "/home/jonathan/Jonathan/programs/PerlScripts";
my @oldNamesArray;
my $oldName; 
my $file;
# Text::Iconv is not really required.
# This can be any object with the convert method. Or nothing.

use Spreadsheet::XLSX;

my $excel = Spreadsheet::XLSX -> new ('test1.xlsx',$converter);

foreach my $sheet (@{$excel -> {Worksheet}}) {
    
    printf("table: %s\n", $sheet->{Name});
    $sheet -> {MaxRow} ||= $sheet -> {MinRow};

    foreach my $row ($sheet -> {MinRow} .. $sheet -> {MaxRow}) {
	$sheet -> {MaxCol} ||= $sheet -> {MinCol};
	
	foreach my $col ($sheet -> {MinCol} ..  $sheet -> {MaxCol}) {
	    my $cell = $sheet -> {Cells} [$row] [$col];
	    if($col== 2)
	    { 
		if ($cell) {
		    #printf("( %s , %s ) => %s", $row, $col, $cell -> {Val});
		    my $tmpName = $cell->{Val};
		    my @strArray = split(//, $tmpName);
		    $oldName = "";
		    foreach $char(@strArray)
		    {
			if ( $char =~ /[a-zA-Z0-9]/) {
			    $oldName = $oldName.$char;
			}
		    }

		}
		$cell = $sheet -> {Cells} [$row] [$col+1];
		if ($cell) {
		    $newName = $cell->{Val};
		    print("oldName: ",$oldName,"\n");
		    print("newName: ",$newName);
		    push(@oldNamesArray,$oldName)
		}
		else {
		    print("COULD NOT FIND NEW FILENAME FOR IT");
		}
	    }
	}	
    }    
 }

# opendir(BIN, $docsFolder) or die "Can't open $docsFolder: $!";
# while( defined ($file = readdir BIN) )
# {
#         if($file =~ m/pdf/)
#         {
# 	    print("filename: ",$file,"\n");
# 	    foreach my $name (@oldNamesArray)
# 	    {
# 		my $nameLength = length($name); 
# 		my $testName = substr($name,0,15);
# 		my $testFile = substr($file,0,15);
# 		print("comparing: ",$testName," and \n ",$testFile,"\n");
# 		if($testName =~ $testFile)
# 		{
# 		    print("found it!: ",$file,'\n');
		    
# 		}
# 	    }
# 	    #rename($file,"moo.txt");
#         }
# }
# closedir(BIN);

