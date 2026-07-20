import java.lang.reflect.*;
import com.actelion.research.chem.reaction.ReactionEncoder;
public class InspectReactionEncoder {
  public static void main(String[] args) throws Exception {
    for (Method m : ReactionEncoder.class.getDeclaredMethods()) {
      if (m.getName().startsWith("decode") || m.getName().startsWith("encode")) System.out.println(m.toString());
    }
  }
}
