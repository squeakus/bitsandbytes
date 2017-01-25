#!/usr/bin/perl -w

package RGEVA::ExperimentManager;

use strict;
use threads; 
use Thread::Queue;

my $DataQueue = Thread::Queue->new; 
#Number of threads
my $nThreads;
my $exp_id;
my $nRuns;
my $GEVA_DIR;
if (@ARGV == 4) {
  ($nThreads, $exp_id, $nRuns, $GEVA_DIR) = @ARGV;
} else {
  print "Usage: $0 nThreads exp_id nRuns GEVA_DIR\n";
  exit(0);
}
#Trim / from GEVA_DIR
$GEVA_DIR=~s/\/$//;

#Experiments
my $CLASS_PATH = "Main.HelloWorld";
my $java_cmd = "java -server -Xmx448m -cp $CLASS_PATH -jar $GEVA_DIR/bin/GEVA.jar";
my $properties_file = "-properties_file $GEVA_DIR/param/Parameters/ExperimentBase.properties";
my $population_size = "-population_size 500";
my $generations = "-generations 50";
my $elite_size = "-elite_size 2";

my $XO_OP = "crossover_operation";
my $XO_PR = "crossover_probability";
my $M_OP = "mutation_operation";
my $M_PR = "mutation_probability";
my $FIT_FUN = "fitness_function";
my $GRAMMAR_FILE = "grammar_file";
my %experiments = 
  (
   "wm" => "-$FIT_FUN FitnessEvaluation.PatternMatch.WordMatch" .
   " -$GRAMMAR_FILE $GEVA_DIR/param/Grammar/letter_grammar.bnf" .
   " -word experimental",
   "sr" => "-$FIT_FUN FitnessEvaluation.SymbolicRegression.SymbolicRegressionJScheme".
   " -$GRAMMAR_FILE $GEVA_DIR/param/Grammar/sr_grammar_sch.bnf",
   "efp" => "-$FIT_FUN FitnessEvaluation.ParityProblem.EvenFiveParityFitnessBSF" .
   " -$GRAMMAR_FILE $GEVA_DIR/param/Grammar/efp_grammar_gr.bnf",
   "sf" => "-$FIT_FUN FitnessEvaluation.SantaFeAntTrail.SantaFeAntTrailBSF" .
   " -$GRAMMAR_FILE $GEVA_DIR/param/Grammar/sf_grammar_gr.bnf",
   "paint" => "-$FIT_FUN FitnessEvaluation.Canvas.Paint" .
   " -$GRAMMAR_FILE $GEVA_DIR/param/Grammar/paint.bnf" 
    );
my %xo_ops =
  (
   "xo_std" => "-$XO_OP Operator.Operations.SinglePointCrossover" .
   "-fixed_point_crossover false",
   "xo_cs" => "-$XO_OP Operator.Operations.ContextSensitiveCrossover"
  );
my %xo_prs =
  (
   "xo_0.1" => "-$XO_PR 0.1",
   "xo_0.5" => "-$XO_PR 0.5",
   "xo_0.9" => "-$XO_PR 0.9"
  );
my %m_ops =
  (
   "m_std" => "-$M_OP Operator.Operations.IntFlipMutation",
   "m_cs" => "-$M_OP Operator.Operations.ContextSensitiveMutation"
  );
my %m_prs =
  (
   "m_0.01" => "-$M_PR 0.01",
   "m_0.1" => "-$M_PR 0.1",
   "m_0.9" => "-$M_PR 0.9"
  );
# Create experiments
my %variants;
# Create dir
if(! -d $exp_id) {
    mkdir($exp_id) or die("Cannor mkdir $exp_id $!\n");
  }
foreach my $experiment (keys %experiments) {
  foreach my $xo_op (keys %xo_ops) {
    foreach my $xo_pr (keys %xo_prs) {
      foreach my $m_op (keys %m_ops) {
	foreach my $m_pr (keys %m_prs) {
	  #Create dir
	  my $id_str = join("_",$exp_id,$experiment,$xo_op,$xo_pr,$m_op,$m_pr);
	  my $var_dir = $exp_id . "/" . $id_str;
	  if(!-d $var_dir) {
	    mkdir($var_dir) or die("Cannor mkdir $var_dir $!\n");
	  }
	  #output
	  my $output = "-output $var_dir/";
	  my $stdRedirect = "> " . $id_str . ".out";
	  # defined outside properties filee
	  my $pl_defs = join(" ", $properties_file, $population_size, $generations, $elite_size);
	  #variants
	  $variants{$id_str} = join(" ",$java_cmd, $pl_defs, $experiments{$experiment}, $output, $xo_ops{$xo_op}, $xo_prs{$xo_pr}, $m_ops{$m_op}, $m_prs{$m_pr}, $stdRedirect);
	}
      }
    }
  }
}

#Create threads
my @threads;
for(my $i=0;$i<$nThreads;$i++) {
    my $thr = threads->new(
	sub { 
	    while (my $DataElement = $DataQueue->dequeue) { 
		print "$DataElement\n";
		print `$DataElement\n`;
	    } 
	}
    );
    push(@threads, $thr);
}

# Enque variants
foreach my $variant (keys %variants) {
    for(my $i=0;$i<$nRuns;$i++) {
	$DataQueue->enqueue($variants{$variant}); 
    }
}
# Allow threads to end
foreach my $thr (@threads) {
    $DataQueue->enqueue(undef);
}
# Finish threads
foreach my $thr (@threads) {
    $thr->join;
    print "Finisihing thread $thr\n";
}
exit(0);
