#!/usr/bin/perl
    use strict;
    use warnings;
 
    #this program showed me to use chomp to remove the newline part
    #which screws up equality

    my $string2 = "y";
    
    print("Do you want this program to remove results as well as pad them?(Y/N)");
	my $string1 = <STDIN>;
	 chomp($string1);
     print("This is what you entered: ".$string1."\n");
	 print("To compare against: ".$string2."\n");
    if ($string1 eq $string2) {
        print "Equal\n";
    } else {
        print "Not equal\n";
    }
