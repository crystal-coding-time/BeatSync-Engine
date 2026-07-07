# Design: intent-based GUI reorganization

Status: **being implemented** (wave 9, 2026-07-07). Owner direction: *"The interface should
reflect this automated intelligence. Hide the low-level mechanical choices (like 'Stretch,'
which should rarely be used anyway) and expose high-level creative intents."*

Owner decisions on the proposal: Stretch **removed from the UI** (engine `'stretch'` value
kept for headless/settings compatibility); preset picker **rejected** (that section is not
implemented); processing mode demoted to Advanced per recommendation.

## Goal

The engine now makes most mechanical decisions itself: the three-tier fit ladder
(subject-tracked smart crop ≤15% → graded echo fill 15–40% → scanning pan >40%),
automatic split-screen pairing (on by default), beat-aware curated effects, source-coverage
variety, and a fixed 16:9 1080p canvas. The GUI still presents **15 style/settings widgets**
in one flat column, most of which duplicate decisions the engine already makes well.
Reorganize into three tiers — creative intent always visible, mechanical knobs in one
collapsed Advanced accordion, fully-automated choices gone — **without changing a single
internal value or function signature**.

## Inherited constraints (non-negotiable)

- **Zero engine changes.** This is a `create_ui()` layout pass; every widget keeps its
  variable name, internal values, and position in the `process_btn` inputs list.
- The `inputs=[...]` list (gui.py:938–948) ↔ `process_video` positional signature
  (gui.py:637–646) coupling stays byte-for-byte compatible — verified by the AST check
  (see verification plan).
- Headless callers of `_process_video_impl` (gui.py:398–416) and the settings-dict
  override (gui.py:421–435) keep working unchanged.
- The dropzone workaround cluster (gradio#10325, gui.py:788–837) and the disable-button
  render chain (gui.py:928–955) move as opaque units or not at all.
- Windows parity per CLAUDE.md: `gui.py` is shared (NVENC/VideoToolbox/CPU branch at
  gui.py:909–914); values like `'stretch'` stay *accepted by the engine* even if hidden.

## 1. Current-state inventory

Every user-facing control in `create_ui()` (src/gui.py). Classification:
**(a)** creative intent · **(b)** mechanical knob the engine could decide · **(c)** plumbing.

| Control (var) | Widget | Default | Class | gui.py | Notes |
|---|---|---|---|---|---|
| `audio_input` | File | — | c | 780 | uppercase ext variants for gradio#10746 |
| `video_dropzone` | File (multiple) | — | c | 792 | value never persists (gradio#10325); real input is `video_state` |
| `video_state` | State | `[]` | c | 793 | **this**, not the dropzone, feeds `process_video` |
| `video_list` + remove/clear btns | CheckboxGroup + 2 Buttons | hidden | c | 794–797 | file management for the dropzone workaround |
| `custom_fps` | Number | `None` (auto) | b | 841 | engine already follows highest-res source (gui.py:493–510) |
| `output_format_input` | Dropdown | `16:9_1080p` | **a** | 845 | canvas = a real creative/destination choice |
| `fit_mode_input` | Radio (crop/blur/pad/stretch) | `crop` | b (pad is a-ish) | 849 | `crop` *is* the smart ladder; blur/pad force one tier; stretch = pre-Phase-1.5 legacy distortion |
| `effect_style_input` | Radio (clean/amv/hype) | `clean` | **a** | 853 | the calm-cinematic ↔ hype-AMV axis already exists here |
| `effect_intensity_input` | Slider 0–1 | `0.7` | b | 857 | curated recipes are tuned around 0.7 |
| `effect_mode_input` | Radio (curated/custom/shuffle) | `curated` | b | 859 | power-user feature; drives palette/seed visibility (gui.py:883–893) |
| `effect_palette_input` | CheckboxGroup | `[]`, hidden | b | 863 | visible only in custom mode |
| `effect_seed_input` | Number | `0`, hidden | b | 867 | visible only in shuffle mode |
| `look_input` | Dropdown | `''` (None) | **a** | 870 | color grade = pure creative intent; values are `.cube` paths from `list_looks()` |
| `variety_input` | Slider 0–1 | `0.4` | b | 873 | default already encodes the right policy |
| `speed_ramps_input` | Checkbox | `False` | b | 876 | experimental, opt-in by design |
| `split_screen_input` | Checkbox | `True` | b | 879 | engine decides when to fire; toggle is an opt-*out* |
| `text_entries_input` | Textbox (4 lines) | `''` | **a** | 897 | the user's words — cannot be automated |
| `text_position_input` | Radio | `bottom` | a-minor | 902 | sensible default; rarely changed |
| `text_scale_input` | Slider 0.5–2 | `1.0` | a-minor | 905 | sensible default; rarely changed |
| `processing_mode` | Radio (platform-dependent) | HW encoder | b/c | 909–914 | quality/export pipeline choice; ProRes is a deliberate pro workflow |
| `output_filename` | Textbox | `music_video.mp4` | c | 919 | timestamp appended anyway |
| `process_btn` | Button | — | c | 921 | disable/re-enable chain at 928–955 |

Score: **4 true creative intents** (canvas, style, look, text) out of 15 non-plumbing
controls. Everything else is a knob whose default the engine already gets right.

## 2. Proposed surface

Design rule: **layout-only change.** Widgets move between visual containers; nothing about
their identity changes. Gradio wires events by component reference, not screen position, so
the `process_btn.click(...).then(inputs=[...])` list is untouched.

### Tier 1 — always visible

| Survivor | Why it stays up front |
|---|---|
| Audio + video inputs, loaded-videos list | plumbing; nothing to do without them |
| **Preset** picker (new, see §3) | one-click intent bundles; pure UI sugar |
| **Style** (`effect_style_input`, relabeled) | this *is* the calm-cinematic ↔ hype-AMV energy axis the owner asked for. Relabel choices only: `Minimal (clean cuts)` / `Music video (AMV)` / `Hype` — values stay `clean`/`amv`/`hype` |
| **Look** (`look_input`) | color grade is the single biggest "feel" lever; free at render time |
| **Output canvas** (`output_format_input`) | destination (YouTube vs Shorts) is a decision only the user can make |
| **Text entries** (`text_entries_input`) | user-authored content; can't be automated |
| Process button + status/preview column | plumbing |

Deliberately *not* Tier 1: effect intensity (curated recipes are tuned around 0.7 — surfacing
it invites fiddling that makes results worse), text position/size (good defaults, one click
away), processing mode (the auto-picked HW encoder is right for ~all renders).

### Tier 2 — one "Advanced" accordion, `open=False`

One `gr.Accordion('⚙️ Advanced', open=False)` containing, grouped:

- **Effects**: `effect_mode_input` + `effect_palette_input` + `effect_seed_input`
  (the existing `.change` visibility handler at gui.py:883–893 works unchanged inside an
  accordion — visibility updates are independent of container), `effect_intensity_input`.
- **Editing**: `variety_input`, `speed_ramps_input`, `split_screen_input`.
- **Framing**: `fit_mode_input`, reduced to **three** choices:
  - `Auto (smart)` → `'crop'` — the ladder, default. Rename because "Smart crop" undersells
    it: at >15% mismatch it isn't a crop at all.
  - `Blurred background` → `'blur'` — forces the echo-fill tier always; a legitimate
    deliberate look (the lyric-video idiom) distinct from what Auto would choose per clip.
  - `Letterbox` → `'pad'` — a legitimate deliberate style (festival/cinema framing).
  - **Stretch: removed from the UI, value kept in the engine.** Case: `'stretch'` exists
    only as pre-Phase-1.5 parity (upstream Windows behavior before this fork's fit engine —
    ROADMAP Phase 1.5). CLAUDE.md's parity rule protects Windows *code paths*, not GUI
    choices, and upstream `main` doesn't have this radio at all — it was added on
    `mac-port`. Hiding vs deleting: **hide in UI, keep the engine value** is the low-risk
    move. `get_fit_filters` keeps handling `'stretch'`, so headless callers, the settings
    dict, and any reproduction of an old render still work; only the radio option
    disappears. Deleting the engine path would break `_process_video_impl(fit_mode=
    'stretch')` callers for zero gain. If someone truly needs it: settings dict.
- **Output**: `processing_mode` (+ its ProRes caveat markdown, gui.py:915), `custom_fps`,
  `output_filename`, `text_position_input`, `text_scale_input`.

### Tier 3 — removed from the UI (engine-decided)

- **Stretch** fit option (see above) — the only widget *choice* that dies. No whole widget
  is removed: everything else already earns its Advanced slot as a deliberate override of
  a good automatic decision.
- Already engine-decided with no widget today (list for the record, so nobody re-adds
  them): fit-tier selection per clip, split-screen firing/frequency/partner choice,
  effect placement on the beat grid, per-source coverage seating, output FPS
  (highest-res source), GPU mode, worker counts.

### Mockup

```
┌─ 🎵 BeatSync Engine ──────────────────────────────┬──────────────────────┐
│ ### 📁 Input Files                                │ ### 📺 Output        │
│ [ Audio file                              ]       │ [ Status box       ] │
│ [ Video dropzone (always droppable)       ]       │ [                  ] │
│ [ 🎬 Loaded videos (7)  ▢▢▢▢  🗑 ♻️ ]              │ [ Video preview    ] │
│                                                   │ [                  ] │
│ ### 🎨 Create                                     │ [                  ] │
│ Preset:  (•) YouTube montage ( ) Shorts ( ) Cinematic ( ) Custom        │
│ Style:   (•) Minimal  ( ) Music video  ( ) Hype   │                      │
│ Look:    [ None                      ▾]           │                      │
│ Canvas:  [ 16:9 · 1080p (1920×1080)  ▾]           │                      │
│ Text:    [ one entry per line…        ]           │                      │
│          [ @1:23 pins to a timestamp  ]           │                      │
│                                                   │                      │
│ ▸ ⚙️ Advanced                        (collapsed)  │                      │
│ ┆  Effects: mode ○curated ○custom ○shuffle        │                      │
│ ┆           intensity ────●──── 0.7               │                      │
│ ┆  Editing: variety ──●────── 0.4                 │                      │
│ ┆           ☐ Speed ramps   ☑ Pair vertical clips │                      │
│ ┆  Framing: (•) Auto (smart) ( ) Blur bg ( ) Letterbox                  │
│ ┆  Output:  processing mode ○HW ○CPU ○ProRes      │                      │
│ ┆           custom FPS [    ]  filename [music_video.mp4]               │
│ ┆           text position ○lower ○center ○top  size ──●── 1.0           │
│                                                   │                      │
│ [ 🎬 Create Music Video ]                         │                      │
└───────────────────────────────────────────────────┴──────────────────────┘
```

## 3. Preset concept — **recommended: adopt**

A `gr.Radio` ("Preset") in Tier 1 whose `.change` handler emits `gr.update(value=...)` to
the individual controls. Key property: **presets are pure UI sugar** — they set widget
values, the widgets remain the source of truth, and the process chain is untouched (no new
entry in the inputs list; the preset radio is *not* an input to `process_video`). Any
control changed after picking a preset simply overrides it. v1 keeps this dumb: picking a
preset stamps values once; we do **not** track divergence or auto-flip the radio to
"Custom" (that needs `.change` listeners on every stamped control — deferred, see decision
point 4).

| Preset pins → | canvas | style | intensity | look | variety | ramps | split | text pos/size | fit |
|---|---|---|---|---|---|---|---|---|---|
| **YouTube montage** | `16:9_1080p` | `amv` | 0.7 | None | 0.4 | off | on | bottom / 1.0 | crop |
| **Shorts / Reels** | `9:16_portrait` | `hype` | 0.8 | None | 0.5 | off | on (stacks landscape pairs) | center / 1.3 | crop |
| **Cinematic** | `16:9_1080p` | `clean` | 0.7 (moot) | *Warm* | 0.3 | off | on | bottom / 1.0 | crop |
| **Custom** | stamps nothing (escape hatch / initial value) | | | | | | | | |

Not pinned by any preset: files, output filename, custom FPS, processing mode, effect
mode/palette/seed (curated assumed), text entries. Look values are **`.cube` paths**
(gui.py:872, `list_looks()` in src/looks.py:52–60), so the Cinematic preset must resolve
"Warm" by label from `list_looks()` at UI-build time — never hardcode a path — and fall
back to None if the cube is missing.

Why for: the three bundles map exactly onto real destinations, they make the Tier-1 surface
self-explanatory, and the implementation is ~40 lines with zero coupling risk. Why the
counterargument fails: "presets hide state" — they don't here, every pinned value is
visible in its (possibly collapsed) widget immediately after stamping.

## 4. Migration & compatibility

| Concern | Finding |
|---|---|
| `inputs` list ↔ positional args | The fragile coupling (gui.py:938–948 ↔ 637–646) is **unchanged by design** — moving widgets between containers doesn't touch the list. The redesign adds no inputs (preset radio is handled by its own `.change`, not by `process_video`). Run the AST check anyway (below). |
| Headless / smoke callers | `_process_video_impl` kwargs (gui.py:398–416) and the settings-dict-wins override (421–435) are untouched. `fit_mode='stretch'` keeps working because the engine value survives. |
| Settings dict | `render_settings` (gui.py:655–670) keys unchanged; `resolve_target_resolution` already tolerates unknown canvas keys (video_processor.py:249–258). |
| Internal values | All choice values frozen: `crop/blur/pad/stretch`, `clean/amv/hype`, `curated/custom/shuffle`, canvas keys, look paths. Only display labels and container placement change. |
| Windows parity | `create_ui()` is shared; the NVENC/VideoToolbox/CPU branch (gui.py:909–914) moves into the Advanced accordion intact. Upstream `main` merges will conflict in `create_ui()` regardless of this change — the reorg doesn't make that materially worse, and no `os.name` branches are touched. |
| **Cannot move (workaround-locked)** | The dropzone cluster — `video_dropzone`, `video_state`, `video_list`, remove/clear buttons and their three handlers (gui.py:788–837) — must move as one atomic block; `video_state` (not the dropzone) is the process input. The uppercase-extension file filter (gui.py:781–786) is baked at page load (gradio#10746 + README.mac.md note): any filter edit needs a browser refresh, worth re-stating in README. The disable→process→enable chain (gui.py:928–955) is order-critical (re-enable runs on success *and* error). |
| Gradio constraints found | `effect_mode_input.change` visibility toggling (gui.py:883–893) works inside an accordion — `gr.update(visible=...)` is container-independent. `gr.Accordion(open=False)` is stock Blocks; no CSS needed beyond the existing `STATUS_BOX_CSS`. |
| Saved user workflows | The app persists nothing between sessions (no settings file; `session_state` is per-tab), so no stored config can break. The only "workflow" at risk is muscle memory + any external script calling `_process_video_impl` — both preserved. |

## 5. Implementation wave plan

| Wave | Files | Est. | Notes |
|---|---|---|---|
| **A. Layout + labels** | `src/gui.py` (`create_ui()` only), `src/ui_content.py` (labels/info strings) | ~150 diff lines | Tier groups, Advanced accordion, Stretch removal, Style relabel. One agent; no engine files. |
| **B. Presets** | `src/gui.py` | ~50 lines | Preset radio + `.change` stamping handler; label-resolved look. Sequenced *after* A (same file, same function) — not parallel. |
| **C. Docs** | `README.mac.md` ("Style controls" restructured to mirror the tiers), `docs/ROADMAP.md` (new phase entry) | ~60 lines | Can run parallel with B (disjoint files) but reads better written last. |

Single-agent-per-wave; A→B sequenced (both edit `create_ui()`), C parallel with B or after.
Same rollout as always: land uncommitted → restart service → owner tests → commit.

**Verification plan**
1. **AST widget↔param check**: parse `src/gui.py` with `ast`, locate the `.then(fn=
   process_video, inputs=[...])` call, extract the input variable names in order, and
   assert they match `process_video`'s positional parameter names one-for-one (with
   `video_state`↔`video_files` and `session_state` as the known aliases). Must pass before
   and after; the diff of the two runs must be empty.
2. **Headless smoke**: `PYTHONPATH=src BEATSYNC_DISABLE_QWEN=1`, synthetic beat.wav +
   testsrc clips → `gui._process_video_impl(...)` twice — once with defaults, once with
   `fit_mode='stretch'` in the settings dict (proves the hidden value still renders).
3. **Screenshot-by-render**: launch the UI, screenshot Tier 1 collapsed and Advanced
   expanded; eyeball that the effect-mode palette/seed visibility toggling still works
   inside the accordion; click each preset and confirm the stamped values.
4. One full GUI render on real footage (owner) before commit, per the working rhythm.

## 6. Decision points for the owner

1. **Stretch** — remove from the UI but keep `'stretch'` working in the engine
   (recommended), keep it as a fourth radio in Advanced, or delete the engine path too?
   Recommendation: UI-remove only; deleting the engine path breaks headless callers for
   no benefit.
2. **Presets** — adopt the 3-preset picker (recommended) or skip and rely on tiering
   alone? If adopted: are YouTube montage / Shorts / Cinematic the right three, and are
   the pinned values in §3 right (esp. Cinematic = Warm look)?
3. **Processing mode placement** — Advanced (recommended; the auto-picked HW encoder is
   right virtually always and ProRes users know to look) or keep in Tier 1? ProRes has a
   real behavioral footprint (no effects/text/looks/ramps), which argues for *findable*,
   not *prominent*.
4. **Preset divergence indicator** — v1 stamps values and forgets (recommended), or track
   changes and flip the radio to "Custom" when any pinned control moves (+~30 lines of
   listeners, more Gradio event surface)?
5. **Style labels** — rename to `Minimal / Music video / Hype` (recommended; values
   unchanged) or keep `Clean / AMV / Hype`?
6. **Effect intensity** — stay in Advanced (recommended; recipes are tuned around 0.7) or
   promote to Tier 1 as the "how much" companion to Style?
