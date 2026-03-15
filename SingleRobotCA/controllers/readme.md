tick():
  1. Hard stop   (min_cone < 0.5m)  → always STOP, no release until clear
  2. Score stop  (max_score > 0.35) → STOP, enter danger state
  3. Hold        (danger active AND min_cone < 1.5m) → stay STOP silently
  4. Clear       (score < 0.15 AND min_cone > 1.5m) → FAILURE → navigate
