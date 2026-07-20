import java.io.*;
import com.actelion.research.chem.reaction.Reaction;
import com.actelion.research.chem.reaction.ReactionEncoder;
import com.actelion.research.chem.IsomericSmilesCreator;

public class ConvertRxnSimple {
  public static void main(String[] args) throws Exception {
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    String line;
    int count = 0;
    while ((line = br.readLine()) != null && count < 8) {
      if (line.startsWith("<") || line.startsWith("RxnMapping") || line.trim().isEmpty()) continue;
      String[] p = line.split("\\t", -1);
      if (p.length < 15) continue;
      for (boolean keep : new boolean[]{false, true}) {
        try {
          Reaction rxn = ReactionEncoder.decode(p[7], keep);
          String smi = IsomericSmilesCreator.createReactionSmiles(rxn);
          System.out.println("OK" + keep + "\t" + p[9] + "\t" + p[11] + "\t" + p[12] + "\t" + p[13] + "\t" + smi);
        } catch (Throwable t) {
          System.out.println("ERR" + keep + "\t" + p.length + "\t" + t.getClass().getSimpleName() + "\t" + t.getMessage());
        }
      }
      count++;
    }
  }
}
