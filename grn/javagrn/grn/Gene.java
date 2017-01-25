package grn;

public class Gene {

  /** */
  public static int TF_PROMOTER = 0x00000000;

  /** */
  public static int P_PROMOTER = 0x000000FF;

  /** */
  public static int SIZE = 8;

  /** */
  public int enhancer;

  /** */
  public int inhibitor;

  /** */
  public int promoter;

  /** */
  public int codons[];

  /**
   *
   */
  public Gene() {
    ;
  }

}