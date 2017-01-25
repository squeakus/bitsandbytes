package grn;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Random;

import grn.helper.ArrayUtils;
import Individuals.GEChromosome;

public class Grn {
  public static int runcount = 1;	
  public static final double ZERO = 1e-10;

  private static int INIT_PERIOD = 10000;

  private static int REST_STEP = 1;

  private static double REST_DELTA = ZERO;

  public double[][] initResults;
  public double[][] results;

  public GEChromosome chromosome;
  public Gene[] genes;
  public Gene[] p_genes;
  public Protein[] proteins;
  public Protein[] p_proteins;
  public double input_concentration;
  public int number_of_inputs;
  public int umax;

  /**
   *
   */
  public Grn(int[] codons) {
    this(codons, new ArrayList<Protein>());
  }

  /**
   *
   */
  public Grn(GEChromosome chromo, ArrayList<Protein> input_proteins) {
    BitScanner hunter = new BitScanner(chromo);
    genes = hunter.getTFGenes();
    p_genes = hunter.getPGenes();
    hunter = null;

    proteins = ProteinProducer.expressGenes(genes);
    p_proteins = ProteinProducer.expressGenes(p_genes);

    calculateInputConcentration(input_proteins);
    setInitialProteinConcentrations();

    number_of_inputs = input_proteins.size();
    proteins = ArrayUtils.resizeArray(proteins, proteins.length+number_of_inputs);
    for (int p = number_of_inputs; p > 0; p--)
      proteins[proteins.length-p] = input_proteins.get(p-1);
  }
  
  /** 
   *
   */
  public Grn(int[] codons, ArrayList<Protein> input_proteins) {
    BitScanner hunter = new BitScanner(codons);
    genes = hunter.getTFGenes();
    p_genes = hunter.getPGenes();
    hunter = null;

    proteins = ProteinProducer.expressGenes(genes);
    p_proteins = ProteinProducer.expressGenes(p_genes);

    calculateInputConcentration(input_proteins);
    setInitialProteinConcentrations();

    number_of_inputs = input_proteins.size();
    proteins = ArrayUtils.resizeArray(proteins, proteins.length+number_of_inputs);
    for (int p = number_of_inputs; p > 0; p--)
      proteins[proteins.length-p] = input_proteins.get(p-1);
  }

  /**
   *
   */
  public Grn(String codonString, ArrayList<Protein> input_proteins) {
    String[] codonStrings = codonString.split(" ");
    int[] codons = new int[codonStrings.length];
    for (int i = 0; i < codonStrings.length; i++)
      codons[i] = Integer.parseInt(codonStrings[i]);

    BitScanner hunter = new BitScanner(codons);
    genes = hunter.getTFGenes();
    p_genes = hunter.getPGenes();
    hunter = null;

    proteins = ProteinProducer.expressGenes(genes);
    p_proteins = ProteinProducer.expressGenes(p_genes);

    calculateInputConcentration(input_proteins);
    setInitialProteinConcentrations();

    number_of_inputs = input_proteins.size();
    proteins = ArrayUtils.resizeArray(proteins, proteins.length+number_of_inputs);
    for (int p = number_of_inputs; p > 0; p--)
      proteins[proteins.length-p] = input_proteins.get(p-1);
  }

  /**
   * Return an integer array of the integer representations of each
   * gene, and its components. 
   *
   * @return array of codon values encoding all genes
   */
  public int[] getGRNEncoding() {
    int[] codons = new int[(genes.length + p_genes.length)*8];
    int i = 0;
    for (Gene g : genes) {
      codons[i++] = g.enhancer;
      codons[i++] = g.inhibitor;
      codons[i++] = g.promoter;
      for (int j = 0; j < 5; j++)
        codons[i++] = g.codons[j];
    }

    for (Gene g : p_genes) {
      codons[i++] = g.enhancer;
      codons[i++] = g.inhibitor;
      codons[i++] = g.promoter;
      for (int j = 0; j < 5; j++)
        codons[i++] = g.codons[j];
    }
    return codons;
  }

  /**
   *
   */
  private void calculateInputConcentration(ArrayList<Protein> input_proteins) {
    input_concentration = 0.0;
    for (Protein p : input_proteins)
      input_concentration += p.concentration;
  }

  /**
   *
   */
  private void setInitialProteinConcentrations() {
    for (Protein p : proteins)
      p.concentration = (1.0 - input_concentration) / proteins.length;

    for (Protein p : p_proteins)
      p.concentration = 1.0 / p_proteins.length;
  }

  /**
   *
   */
  private void normaliseTFProteinConcentrations() {
    double total = 0;
    for (int i = 0; i < proteins.length-number_of_inputs; i++)
      total += proteins[i].concentration;
      
    if (total > 0.0)
      for (int i = 0; i < proteins.length; i++)
        if (i  < proteins.length-number_of_inputs) {
          proteins[i].concentration *= 1.0 - input_concentration;
          proteins[i].concentration /= total;
        }
  }
  
  /**
   * Checks if the model is at rest.  For each protein, check if over
   * the last REST_STEP time steps the value hasn't changed more than
   * REST_DELTA in concentration.
   *
   * Maybe it would be better to check the delta at each timestep from
   * t-REST_DELTA to t and make sure it's zero'd?
   *
   * @param results the model data
   * @param t the current timestep
   * @result if the model has leveled out
   */
  private boolean atRest(double[][] results, int t) {
    if (t < REST_STEP)
      return false;

    for (int i = 0; i < proteins.length; i++)
      if (Math.abs(results[t][i] - results[t-REST_STEP][i]) > REST_DELTA) {
        return false;
      }

    for (int i = 0; i < p_proteins.length; i++)
      if (Math.abs(results[t][proteins.length+i] - 
                   results[t-REST_STEP][proteins.length+i]) > REST_DELTA) {
        return false;
      }

    return true;
  }

  /**
   * Inject input proteins into the model (replacing the current inputs)
   *
   * @param input_proteins Proteins to replace the current input proteins
   */
  public void injectInputs(ArrayList<Protein> input_proteins) {
    //If wrong size, resize
    if (input_proteins.size() != number_of_inputs)
      proteins = ArrayUtils.resizeArray(proteins, proteins.length-number_of_inputs+input_proteins.size());
    
    // Add the new inputs
    number_of_inputs = input_proteins.size();
    for (int p = number_of_inputs; p > 0; p--)
      proteins[proteins.length-p] = input_proteins.get(p-1);
    calculateInputConcentration(input_proteins);

    // Normalise the rest of the TF concnetrations
    normaliseTFProteinConcentrations();
  }

  /**
   * Run until at rest or t = INIT_PERIOD
   */
  public void init() {
    //initResults = run(INIT_PERIOD, true);
  }
  
  /**
   *
   */
  public double[][] run(int timeSteps, boolean initialising) {
    results = new double[timeSteps+1][proteins.length+p_proteins.length];
    umax = -1;
    for (Gene g : genes) {
      int temp;
      if ((temp = ProteinProducer.calculateUMax(g, proteins)) > umax)
        umax = temp;
    }
    for (Gene g : p_genes) {
      int temp;
      if ((temp = ProteinProducer.calculateUMax(g, proteins)) > umax)
        umax = temp;
    }
    System.out.println("UMAX:"+umax);
    
    int t;
    for (t = 0; t < timeSteps && (initialising ? !atRest(results, t-1) : true); t++) {


      //Record the current state
      for (int i = 0; i < proteins.length; i++) 
        results[t][i] = proteins[i].concentration;
      for (int i = 0; i < p_proteins.length; i++) 
        results[t][proteins.length+i] = p_proteins[i].concentration;

      /* Calculate production rates */
      double[] geneProductionRates = new double[genes.length];
      double[] p_geneProductionRates = new double[p_genes.length];
      for (int i = 0; i < genes.length;  i++)
        geneProductionRates[i] = ProteinProducer.produce(genes[i], proteins[i], proteins, umax);
      for (int i = 0; i < p_genes.length;  i++)
        p_geneProductionRates[i] = ProteinProducer.p_produce(p_genes[i], p_proteins[i], proteins, umax);

      /* Update protein concentrations */
      for (int i = 0; i < genes.length;  i++) {
        proteins[i].concentration += geneProductionRates[i];
        if (proteins[i].concentration < ZERO)
          proteins[i].concentration = ZERO;
      }
      for (int i = 0; i < p_genes.length;  i++) {
        p_proteins[i].concentration += p_geneProductionRates[i];
        if (p_proteins[i].concentration < ZERO)
          p_proteins[i].concentration = ZERO;
      }

      /* Normalise TF concentration levels */
      double total = 0;
      for (int i = 0; i < proteins.length-number_of_inputs; i++)
        total += proteins[i].concentration;
      if (total > 0.0)
        for (int i = 0; i < proteins.length; i++) {
          if (i  < proteins.length-number_of_inputs) {
            proteins[i].concentration *= 1.0 - input_concentration;
            proteins[i].concentration /= total;
          }
      }
      
      /* Normalise P concentration levels */
      total = 0;
      for (Protein p : p_proteins)
        total += p.concentration;
      
      if (total > 0.0)
        for (int i = 0; i < p_proteins.length; i++) {
          p_proteins[i].concentration /= total;
        }
    }

    //Record the current state
    for (int i = 0; i < proteins.length; i++) 
      results[t][i] = proteins[i].concentration;
    for (int i = 0; i < p_proteins.length; i++) 
      results[t][proteins.length+i] = p_proteins[i].concentration;

    return results;
  }

  public static void main(String[] args) {
   for (int t = 0; t < Integer.parseInt(args[0]); t++) {
     Random r = new Random(3);
     
     
     int[] codons = new int[128];
     for (int i = 0; i <  128; i++){
      codons[i] = r.nextInt();
       System.out.print("\""+Integer.toBinaryString(codons[i])+"\",");
     }
     Grn grn = new Grn(codons);
     grn.init();
     ArrayList<Protein> ins = new ArrayList<Protein>();
     ins.add(new Protein(0.05, 0x00000000));
     ins.add(new Protein(0.05, 0x0000FFFF));
     ins.add(new Protein(0.05, 0xFFFF0000));
     ins.add(new Protein(0.05, 0xFFFFFFFF));
     grn.injectInputs(ins);
     GRNPrinter.printGRNToFile("grn_test_run_output_new", grn, grn.run(1000, false));
   }

    // GEChromosome testChromo = new GEChromosome();
    // for (int i = 0; i < args.length; i+=4) {
    //   byte[] bytes = new byte[4];
    //   for (int b = 0; b < 4; b++) 
    //     bytes[b] = Byte.parseByte(args[i+b]);
      
    //   int int32 = intFromByteArray(bytes);
    //   System.out.println("Adding: "+int32);
    //   testChromo.add(int32);
    // }
    // Grn grn = new Grn(testChromo, new ArrayList<Protein>());
    // ArrayList<Protein> ins = new ArrayList<Protein>();
    // ins.add(new Protein(0.05, 0x00000000));
    // ins.add(new Protein(0.05, 0x0000FFFF));
    // ins.add(new Protein(0.05, 0xFFFF0000));
    // ins.add(new Protein(0.05, 0xFFFFFFFF));
    // grn.injectInputs(ins);
    // grn.init();
    // GRNPrinter.printGRNToFile("grn_test_run_output_new", grn, grn.run(1000, false));
  }

  private static int intFromByteArray(byte[] bytes) {
    return (bytes[0] & 0x000000FF) << 24 |
      (bytes[1] & 0x000000FF) << 16 |
      (bytes[2] & 0x000000FF) << 8 |
      (bytes[3] & 0x000000FF);
  }

  public void printState() {
    for (int i = 0; i < proteins.length-number_of_inputs; i++)
      System.out.print("TF"+i+" ");
    for (int i = 0; i < number_of_inputs; i++)
      System.out.print("I"+i+" ");
    for (int i = 0; i < p_proteins.length; i++)
      System.out.print("P" + i + " ");
    System.out.print("\n");

    for (int i = 0; i < proteins.length-number_of_inputs; i++)
      System.out.print(proteins[i].concentration+":"+i+" ");
    for (int i = proteins.length-number_of_inputs; i < proteins.length; i++)
      System.out.print(proteins[i].concentration+" ");
    for (int i = 0; i < p_proteins.length; i++)
      System.out.print(p_proteins[i].concentration+ " ");
    System.out.print("\n");
  }
}
