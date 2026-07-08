# Work tracker

All tracked work is complete. Waves 11-16 (2026-07-08) are committed and documented in
`docs/ROADMAP.md`:

- Wave 11 - review-driven hardening + perf + GIF trailing-frame fix (f2fc6b3)
- Wave 12 - semantic diversity (DINOv2 + windowed MMR) + proportional-fair variety (ed4177e)
- Wave 13 - musical pacing features, loudness-scaled effects, interpolated slow-mo (5f6268a)
- Wave 14 - music-structure understanding: all-in-one-mlx labels + demucs stems (1761969)
- Wave 15 - crossfades, optimal LAP seating, motion smear, 1-frame-segment fix (f13b780)
- Wave 16 - multi-song extended videos, any number of tracks (f13b780)

No open or planned work is tracked here. Workflow for future waves: tracker tasks -> parallel
agents on disjoint files -> integration + docs sync -> headless smoke (double-run framemd5 +
frame guards) -> restart ./run.sh -> owner tests -> commit on his OK.
