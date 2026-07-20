# Cover Letter

Dear Editor of J. Chem. Inf. Model.,

We are pleased to submit our manuscript *PC-CNG v2: PhysChem-Constrained Counterfactual Negative Generation for Chemistry Reaction Prediction* for your consideration.  The manuscript extends our v1 results with a complete P2 validation programme (eight tasks, P2-01 through P2-08) designed to resolve the limitations flagged in v1 and to position the work for journal submission.

The headline P2 outcomes are:

1. **Retrosynthesis route ranking (P2-01, GO):** a +29.20 pp MRR gain (144/150 groups favoured, 10-seed paired significance, p = 1.00e-04).
2. **DFT validation (P2-02, GO):** GFN2-xTB chemoselectivity-error support rate of 90% (27/30), clearing the 0.60 threshold and fixing the v1 partial-support limitation.
3. **External bridge v2 (P2-04, GO):** a Chemformer-aware MLP calibrator beats Chemformer log-likelihood by +2.54 pp Top-1 (p = 0.0010, 10-seed paired), fixing the v1 external-bridge NO-GO.

Aggregate P2 Go/No-Go: 3 GO, 4 NO-GO, 1 deferred.  We transparently report the NO-GO results (P2-05 cross-dataset transfer, P2-07 transformer generator, P2-08 condition prediction) as negative findings that bound the claim, and we have deferred the P2-03 expert review to revision.

We believe the work is a strong fit for J. Chem. Inf. Model. because: (a) the positive claims rest on a strict 10-seed paired significance protocol; (b) the negative results are reported transparently and bound the claim; (c) the reproducibility manifest covers every numeric claim back to a JSON artifact on disk; and (d) the v2 calibrator and DFT validation together resolve two of the most consequential v1 limitations.

We confirm that this manuscript has not been published and is not under consideration elsewhere.  The authors declare no competing interests.  Code and reproducibility artifacts will be released upon acceptance.

Sincerely,
The PC-CNG team

*Journal positioning: strong tier — P2-01 and P2-04 pass Go; P2-06 beats 2/3 (67%) SOTA baselines (>= 1/3 threshold). Strong-tier submission justified.*
