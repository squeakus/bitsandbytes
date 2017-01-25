#!/usr/bin/perl -w

use strict;
use Config::Properties;
use threads; 
use Thread::Queue;
use Cwd;

my $exp_id;
my $results_dir;
my $nRuns;
my $geva_dir;
my $xgrid_dir;
my $prop_file;

if (@ARGV == 1) {
  ($prop_file) = @ARGV;
} else {
  print "Usage: $0 experiment.properties\n";
  exit(0);
}


open PROPS, "< ".$prop_file   
    or die "unable to open configuration file";
  my $properties = new Config::Properties();
  $properties->load(*PROPS);
  $xgrid_dir =$properties->getProperty("xgrid_dir");
  $geva_dir = $properties->getProperty("geva_dir");
  $nRuns =  $properties->getProperty("number_of_runs");
  $exp_id =  $properties->getProperty("experiment_id");


#Trim / from geva_dir 
$geva_dir=~s/\/$//;
$xgrid_dir=~s/\/$//;
print "xgriddir: ".$xgrid_dir."\n";
print "early results: ".$results_dir."\n";
$results_dir = join("/",$xgrid_dir,$exp_id);
print "late results: ".$results_dir."\n";

#Experiments
my $main_class = $properties->getProperty("main_class");
my $java_cmd = "/usr/bin/java -server -Xmx512m -jar $xgrid_dir/bin/GEVA.jar -main_class $main_class";
my $properties_file = join(" ","-properties_file ",$properties->getProperty("properties_file"));
my $population_size = join(" ","-population_size",$properties->getProperty("population_size"));
my $generations = join(" ","-generations",$properties->getProperty("generations"));
my $elite_size = join(" ","-elite_size",$properties->getProperty("elite_size"));
my $max_wraps = join(" ","-max_wraps",$properties->getProperty("max_wraps"));
my $deriv_tree = join(" ","-derivation_tree",$properties->getProperty("deriv_tree"));


my $XO_OP = "crossover_operation";
my $XO_PR = "crossover_probability";
my $M_OP = "mutation_operation";
my $M_PR = "mutation_probability";
my $FIT_FUN = "fitness_function";
my $GRAMMAR_FILE = "grammar_file";

my %experiments = 
  (
   "WMatch" => "-$FIT_FUN FitnessEvaluation.PatternMatch.WordMatch" .
   " -$GRAMMAR_FILE $xgrid_dir/param/Grammar/letter_grammar.bnf" .
   " -word experimental",
    );
my %xo_ops =
  (
   "xo_std" => "-$XO_OP Operator.Operations.SinglePointCrossover" .
   " -fixed_point_crossover false"
  );
my %xo_prs =
  (
   "xo_0.9" => "-$XO_PR 0.9"
  );
my %m_ops =
  (
   "intflip" => "-$M_OP Operator.Operations.IntFlipMutation",
  );
my %m_prs =
  (
   "m_0.02" => "-$M_PR 0.02",
  );
# Create experiments
my %variants;
# Create dir
if(! -d $results_dir) {
	print("making dir $exp_id");
    mkdir($results_dir) or die("Cannot mkdir $exp_id $!\n");
  }
foreach my $experiment (keys %experiments) {
  foreach my $xo_op (keys %xo_ops) {
    foreach my $xo_pr (keys %xo_prs) {
      foreach my $m_op (keys %m_ops) {
	foreach my $m_pr (keys %m_prs) {
	  #Create dir
	  my $id_str = join("_",$exp_id,$experiment,$xo_op,$xo_pr,$m_op,$m_pr);
	  my $var_dir = $results_dir. "/" . $id_str;
	  if(!-d $var_dir) {
	    mkdir($var_dir) or die("Cannot mkdir $var_dir $!\n");
	  }
	  #output
	  my $output = "-output $var_dir/";
	  my $outputFile = join("/",$var_dir,$id_str);
	  my $stdRedirect = ">> " . $outputFile . ".out";
	  # defined outside properties filee
	  my $pl_defs = join(" ", $properties_file, $population_size, $generations, $elite_size, $max_wraps,$deriv_tree);
	  #variants
	  $variants{$id_str} = join(" ",$java_cmd, $pl_defs, $experiments{$experiment}, $output, $xo_ops{$xo_op}, $xo_prs{$xo_pr}, $m_ops{$m_op}, $m_prs{$m_pr}, $stdRedirect);
	}
      }
    }
  }
}

# Writing all the variants out to a file
open (MYBATCHFILE, '>'.$exp_id.'.batch');
foreach my $variant (keys %variants) {
    for(my $i=0;$i<$nRuns;$i++) {
	print MYBATCHFILE $variants{$variant}."\n"; 
    }
}
close (MYBATCHFILE);
exit(0);
