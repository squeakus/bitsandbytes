package grn;

import java.util.ArrayList;

import grn.helper.ArrayUtils;
import Individuals.GEChromosome;

public class BitScanner {

  /** */
  public static int PROMO_MASK = 0x000000FF;

  /** */
  private static int STARTING_INDEX = -(Gene.SIZE - 2)*32;

  /** */
  private int currentIndex;

  /** */
  public int[] chromo;

  public int chromoLength;

  /** */
  Gene[] tfGenes;
  int nTfGenes;

  /** */
  Gene[] pGenes;
  int nPGenes;

  /**
   *
   */
  public BitScanner(GEChromosome chromo) {
    this.chromo = chromo.data;
    chromoLength = chromo.getLength();

    tfGenes = new Gene[chromoLength/8];
    pGenes = new Gene[chromoLength/8];

    currentIndex = STARTING_INDEX;

    findGenes();
  }

  /**
   *
   */
  public BitScanner(int[] codons) {
    this.chromo = codons;
    chromoLength = chromo.length;
    String chromString = "";
    for(int i = 0; i < codons.length; i++){
    	//System.out.print("\""+Integer.toBinaryString(codons[i])+"\",");
    	String bit_string = String.format("%32s", Integer.toBinaryString(codons[i])).replace(' ', '0');
        chromString = chromString.concat(bit_string);
    }
//    System.out.println();
//	System.out.println(chromString);
//	System.out.println("chome length: "+chromString.length());
    tfGenes = new Gene[chromoLength/8];
    pGenes = new Gene[chromoLength/8];

    currentIndex = STARTING_INDEX;

    findGenes();
  }

  /**
   *
   */
  public Gene[] getTFGenes() {
    return tfGenes;
  }

  /**
   *
   */
  public Gene[] getPGenes() {
    return pGenes;
  }

  /**
   *
   */
  private void findGenes() {
    nTfGenes = 0;
    nPGenes  = 0;

    while ((currentIndex = nextPromoter(currentIndex)) > -1) {
      tfGenes = ArrayUtils.checkAndResizeArray(tfGenes, nTfGenes);
      pGenes = ArrayUtils.checkAndResizeArray(pGenes, nTfGenes);

      if (isAPromotor(currentIndex, Gene.TF_PROMOTER)){
    	  System.out.println("found TF promoter at: "+currentIndex);
        tfGenes[nTfGenes++] = getGeneFromPromotorIndex(currentIndex);
      }
      else{
    	  System.out.println("found P promoter at: "+currentIndex); 
    	  pGenes[nPGenes++] = getGeneFromPromotorIndex(currentIndex);
      }	
    }	
    
    tfGenes = ArrayUtils.fitArray(tfGenes, nTfGenes);
    pGenes  = ArrayUtils.fitArray(pGenes, nPGenes);
  }

  /**
   *
   */
  private int getIntFromBitIndex(int index) {
    int codon = index / 32;
    int bitIndex = index % 32;
    
    //    System.out.println(codon+":"+bitIndex);

	if (bitIndex == 0)
		  return chromo[codon];
	else if (codon < chromoLength - 1) {
      // int result = ((chromo[codon] << bitIndex) | (chromo[codon+1] >>> (32 - bitIndex)));
      // System.out.println("c: "+chromo[codon]);
      // System.out.println("c+1: "+chromo[codon+1]);
      // System.out.println("i: "+((chromo[codon] << bitIndex) | (chromo[codon+1] >>> (32 - bitIndex))));

      // for (int i = 0; i < 32; i++) {
      //   int b = result << i;
      //   b = b >>> 31;
      //   System.out.print(b);
      // }
      // System.out.println("");

      return (chromo[codon] << bitIndex) | (chromo[codon+1] >>> (32 - bitIndex));
    }
    else if (codon == chromoLength - 1 && bitIndex == 0)
      return codon;
    else {
      System.out.println("Error getting codon value, index too near end of data: "+index+" -> "+codon+":"+bitIndex+" > "+(chromoLength -1));
      return -1;
    }
  }

  private int getBit(int index) {
    int codon = index / 32;
    int bitIndex = index % 32;
   
    int bit = chromo[codon] << bitIndex;
    return bit >>> 31;
  }

  /**
   *
   */
  private Gene getGeneFromPromotorIndex(int index) {
    Gene g = new Gene();
    g.enhancer  = getIntFromBitIndex(index-64);
    g.inhibitor = getIntFromBitIndex(index-32);
    g.promoter  = getIntFromBitIndex(index);
    g.codons    = new int[5];

    for (int i = 0; i < 5; i++)
      g.codons[i] = getIntFromBitIndex(index + 32 + (i * 32));

    return g;
  }

  /** 
   *
   *
   * @param
   * @param
   * @return
   */
  private int nextPromoter(int prevIndex) {
    if (prevIndex < 0)
      prevIndex = STARTING_INDEX;

    prevIndex += Gene.SIZE*32;

    //System.out.println("Searching for promoter: "+prevIndex+" or "+(prevIndex/32)+":"+(prevIndex%32));
    while (prevIndex <= chromoLength*32 - 192 && 
           !isAPromotor(prevIndex, Gene.TF_PROMOTER)
    		&& !isAPromotor(prevIndex, Gene.P_PROMOTER))
    	prevIndex++;
    	//if (prevIndex > 500 && prevIndex < 600)
    	//	System.out.println(prevIndex+": "+((chromo[prevIndex/32] << (prevIndex%32)) >>> 31));
    //}
    
    if (prevIndex > chromoLength*32 - 192)
      prevIndex = -1;

    //System.out.println("Found promotor at "+prevIndex);
    
    return prevIndex;
  }
  

  /** 
   *
   *
   * @param
   * @param
   * @return
   */
  private boolean isAPromotor(int index, int mask) {
    int i = getIntFromBitIndex(index);

    //if (((i & PROMO_MASK) ^ mask) == 0)
    //System.out.println("HERE: "+i+" "+(i & PROMO_MASK)+" "+mask+" "+index);
    return ((i & PROMO_MASK) ^ mask) == 0;
  }
}
