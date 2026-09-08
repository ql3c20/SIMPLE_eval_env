# Task1–6 HSSD evaluation backgrounds

Included scene entrypoints:
- Task1/4: `102344280/102344280.usd`
- Task2/3: `102344250/102344250_local.usd`
- Task5: `103997919_171031233/103997919_171031233_local.usd`
- Task6: `107734119_175999932/107734119_175999932.usd`

Four scene directories are included (1014 USD/PNG/JPG assets); Task6 thumbnails and the duplicate `(copy).usd` are excluded. Ordinary Git checkout downloads these files; Git LFS is not used for this addition. Keep `props/` and `textures/` alongside each scene.

The published copies of `102344250/102344250.usd` and `102344280/102344280.usd` have stale external props references localized to the included assets. Original working datasets were unchanged. `ICRA2027_MANIFEST.json` records source/export hashes, each path replacement, and recursive dependency validation. Only asset reference paths were rewritten.

All four evaluation entrypoints resolve their USD and texture dependencies within these directories. The remaining `gltf/pbr.mdl` dependency is supplied by Isaac Sim (`kit/mdl/core/mdl/gltf/pbr.mdl` on the source installation). Use the Isaac Sim runtime and its MDL search paths. No rendered-image or GPU closed-loop acceptance is claimed by the file/dependency audit.

Source asset ownership and applicable HSSD asset terms remain unchanged. No policy weights, recordings, or videos are included here.

Task5/6 additions are recorded in `ICRA2027_TASK56_MANIFEST.json` (320 assets). Published scene copies localize stale props/texture paths; Task6 references to `black.usd` are corrected to the existing `props/Black.usd`, whose material is named `black`. Source and published hashes and reference replacements are recorded. Original local assets remain unchanged.
