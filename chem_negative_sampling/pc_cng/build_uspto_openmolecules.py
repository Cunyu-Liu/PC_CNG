"""Convert the OpenMolecules/DataWarrior USPTO subset to reaction SMILES.

The OpenMolecules file ``USPTO_unique_yield25to150.zip`` stores reactions as
DataWarrior/OpenChemLib reaction IDCodes rather than plain SMILES. This helper
uses the OpenChemLib jar in Java source-file mode to decode the ``Reaction``
column and writes a TSV that can be fed into ``pc_cng.data_ingestion``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path


JAVA_SOURCE = r"""
import java.io.*;
import com.actelion.research.chem.reaction.Reaction;
import com.actelion.research.chem.reaction.ReactionEncoder;
import com.actelion.research.chem.IsomericSmilesCreator;

public class DwarToReactionSmiles {
  static String clean(String s) {
    return s == null ? "" : s.replace("\t", " ").replace("\n", " ").replace("\r", " ");
  }

  public static void main(String[] args) throws Exception {
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    System.out.println("source_id\treaction_smiles\tlabel\tyield\tpatent_number\tparagraph_num\tyear\ttext_mined_yield\tcalculated_yield\tdataset_name");
    String line;
    long row = 0, kept = 0, failed = 0;
    while ((line = br.readLine()) != null) {
      if (line.startsWith("<") || line.startsWith("RxnMapping") || line.trim().isEmpty()) continue;
      String[] p = line.split("\\t", -1);
      if (p.length < 15) {
        failed++;
        continue;
      }
      row++;
      try {
        Reaction rxn = ReactionEncoder.decode(p[7], false);
        String smiles = IsomericSmilesCreator.createReactionSmiles(rxn);
        if (smiles == null || smiles.indexOf(">>") < 0) {
          failed++;
          continue;
        }
        kept++;
        String sourceId = "uspto_openmol_" + String.format("%09d", kept);
        String yield = clean(p[12].isEmpty() ? p[13] : p[12]);
        System.out.println(
          sourceId + "\t" + clean(smiles) + "\tpositive\t" + yield + "\t" +
          clean(p[9]) + "\t" + clean(p[10]) + "\t" + clean(p[11]) + "\t" +
          clean(p[12]) + "\t" + clean(p[13]) + "\t" + clean(p[14])
        );
      } catch (Throwable t) {
        failed++;
      }
      if (row % 100000 == 0) {
        System.err.println("processed=" + row + " kept=" + kept + " failed=" + failed);
      }
    }
    System.err.println("done processed=" + row + " kept=" + kept + " failed=" + failed);
  }
}
"""


def find_dwar_member(path: str) -> str:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".dwar"):
                return name
    raise ValueError(f"No .dwar member found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zip", required=True)
    parser.add_argument("--openchemlib-jar", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_tsv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    log_path = args.log or str(Path(args.output_tsv).with_suffix(".log"))
    member = find_dwar_member(args.input_zip)

    with tempfile.TemporaryDirectory() as tmpdir:
        java_path = os.path.join(tmpdir, "DwarToReactionSmiles.java")
        with open(java_path, "w", encoding="utf-8") as handle:
            handle.write(JAVA_SOURCE)

        cmd = ["java", "-cp", args.openchemlib_jar, java_path]
        with zipfile.ZipFile(args.input_zip) as archive, archive.open(member) as source, open(
            args.output_tsv, "wb"
        ) as output_handle, open(log_path, "wb") as log_handle:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=output_handle, stderr=log_handle)
            assert proc.stdin is not None
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                proc.stdin.write(chunk)
            proc.stdin.close()
            return_code = proc.wait()
            if return_code:
                raise RuntimeError(f"OpenChemLib converter failed with exit code {return_code}; see {log_path}")

    with open(args.output_tsv, "rb") as handle:
        line_count = sum(1 for _ in handle)
    summary = {
        "input_zip": args.input_zip,
        "dwar_member": member,
        "openchemlib_jar": args.openchemlib_jar,
        "output_tsv": args.output_tsv,
        "log": log_path,
        "rows": max(0, line_count - 1),
        "label": "positive",
        "source": "uspto_openmolecules_yield25to150",
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
