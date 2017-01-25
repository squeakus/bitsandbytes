package grn;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class GRNPrinter {

  /**
   *
   */
  public static void printGRNToFile(String fileName, Grn grn, double[][] results) {
      //NumberFormat formatter = new DecimalFormat("###.#########");
      NumberFormat formatter = new DecimalFormat("0.000000000");
     try{
       File file =new File(fileName);
       
          //if file doesnt exists, then create it
//       boolean created = false;
//       if(!file.exists()){
         file.createNewFile();
//         created = true;
//       }
       
       //true = append file
       FileWriter fileWritter = new FileWriter(file.getName(),false);
       BufferedWriter bufferWritter = new BufferedWriter(fileWritter);

//       if (created) {
//         for (int i = 0; i < grn.proteins.length-grn.number_of_inputs; i++)
//           bufferWritter.write("TF"+i+" ");
//         for (int i = 0; i < grn.number_of_inputs; i++)
//           bufferWritter.write("I"+i+" ");
//         for (int i = 0; i < grn.p_proteins.length; i++)
//           bufferWritter.write("P"+i+" ");
//         bufferWritter.write("\n");
//       }
       
       for (int t = 0; t < results.length; t++) {
         for (int i = 0; i < results[t].length; i++)
           bufferWritter.write(formatter.format(results[t][i])+" ");
         bufferWritter.write("\n");
       }
       bufferWritter.close(); 
     }
     catch(IOException e){
       e.printStackTrace();
     }   
  }
}
