# Independent boundary review

Use the original audio and manual-review.csv or manual-review.json for the first review. The forms deliberately omit source candidate times. Record the first audible performed phoneme onset at the original sample rate, reviewer identity, and confidence; leave uncertain boundaries ambiguous with a note rather than forcing a single point.

Do not inspect annotation-candidates.json, segments.jsonl, TextGrid, or JSON source timings until independent labels have been recorded. Then compare them to the source candidates. Source annotation agreement is not independent accuracy evidence.

All rows start pending. These samples are unassigned inspection material; they do not establish train/validation/test splits or measured accuracy. Excluded segments and full source validation errors are recorded in excluded-segments.json and quality-summary.json; source annotations are not repaired.
