import java.io.*;
import java.lang.reflect.*;
import com.actelion.research.chem.reaction.Reaction;
import com.actelion.research.chem.reaction.ReactionEncoder;
import com.actelion.research.chem.IsomericSmilesCreator;

public class InspectAndConvert {
  public static void main(String[] args) throws Exception {
    for (Method m : ReactionEncoder.class.getDeclaredMethods()) {
      if (m.getName().equals("decode")) System.err.println(m.toString());
    }
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    String line;
    int count = 0;
    while ((line = br.readLine()) != null && count < 5) {
      if (line.startsWith("<") || line.startsWith("RxnMapping") || line.trim().isEmpty()) continue;
      String[] p = line.split("\\t", -1);
      if (p.length < 15) continue;
      try {
        Reaction rxn = ReactionEncoder.decode(p[7], p[1], p[0], null, null, false);
        String smi = IsomericSmilesCreator.createReactionSmiles(rxn);
        System.out.println("OK\t" + p[9] + "\t" + p[11] + "\t" + p[12] + "\t" + p[13] + "\t" + smi);
      } catch (Throwable t1) {
        try {
          Reaction rxn = ReactionEncoder.decode(p[7], true);
          String smi = IsomericSmilesCreator.createReactionSmiles(rxn);
          System.out.println("OK2\t" + p[9] + "\t" + smi);
        } catch (Throwable t2) {
          System.out.println("ERR\t" + p.length + "\t" + t1.getClass().getSimpleName() + ":" + t1.getMessage() + "\t" + t2.getClass().getSimpleName() + ":" + t2.getMessage());
        }
      }
      count++;
    }
  }
}
