import java.io.*;
import com.actelion.research.chem.reaction.Reaction;
import com.actelion.research.chem.reaction.ReactionEncoder;
import com.actelion.research.chem.IsomericSmilesCreator;

public class DwarToReactionSmiles {
  static String clean(String s) { return s == null ? "" : s.replace("\t", " ").replace("\n", " ").replace("\r", " "); }
  public static void main(String[] args) throws Exception {
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    System.out.println("source_id\treaction_smiles\tpatent_number\tparagraph_num\tyear\ttext_mined_yield\tcalculated_yield\tdataset_name");
    String line;
    long row = 0, kept = 0, failed = 0;
    while ((line = br.readLine()) != null) {
      if (line.startsWith("<") || line.startsWith("RxnMapping") || line.trim().isEmpty()) continue;
      String[] p = line.split("\\t", -1);
      if (p.length < 15) { failed++; continue; }
      row++;
      try {
        Reaction rxn = ReactionEncoder.decode(p[7], false);
        String smi = IsomericSmilesCreator.createReactionSmiles(rxn);
        if (smi == null || smi.indexOf(">>") < 0) { failed++; continue; }
        kept++;
        String sourceId = "uspto_openmol_" + String.format("%09d", kept);
        System.out.println(sourceId + "\t" + clean(smi) + "\t" + clean(p[9]) + "\t" + clean(p[10]) + "\t" + clean(p[11]) + "\t" + clean(p[12]) + "\t" + clean(p[13]) + "\t" + clean(p[14]));
      } catch (Throwable t) {
        failed++;
      }
      if (row % 100000 == 0) System.err.println("processed=" + row + " kept=" + kept + " failed=" + failed);
    }
    System.err.println("done processed=" + row + " kept=" + kept + " failed=" + failed);
  }
}
