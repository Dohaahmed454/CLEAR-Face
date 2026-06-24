# Label Mapping Plan

| Original dataset label | CLEAR-Face label | Keep/merge/remove | Notes |
|---|---|---|---|
| happy | happy | keep | Direct mapping (present in AffectNet & RAF-DB) |
| sad | sad | keep | Direct mapping (present in both datasets) |
| angry | angry | keep | Direct mapping (consistent across datasets) |
| neutral | neutral | keep | Direct mapping (stable baseline class) |
| surprise | surprise | keep | Present in both datasets with sufficient samples |
| disgust | disgust | keep | Available in both datasets; moderate class imbalance may exist |
| fear | fear | keep | Present in both datasets but lower frequency |

## Label decision rule

All labels are included only if they satisfy the following conditions:
1. They exist in both the main dataset (AffectNet) and the external dataset (RAF-DB).
2. They have sufficient number of samples to support stable training and evaluation.
3. They have consistent semantic meaning across datasets.

No label is merged or removed in the final design, ensuring a unified 7-class emotion space suitable for cross-dataset generalization and CARE-based decision evaluation.