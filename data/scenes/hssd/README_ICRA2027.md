# Task1–4 HSSD evaluation backgrounds

Included scene entrypoints:
- Task1/4: `102344280/102344280.usd`
- Task2/3: `102344250/102344250_local.usd`

Both complete local scene directories are included (694 USD/PNG/JPG assets). Ordinary Git checkout downloads these files; Git LFS is not used for this addition. Keep `props/` and `textures/` alongside each scene.

The published copies of `102344250/102344250.usd` and `102344280/102344280.usd` have stale external props references localized to the included assets. Original working datasets were unchanged. `ICRA2027_MANIFEST.json` records source/export hashes, each path replacement, and recursive dependency validation. Only asset reference paths were rewritten.

Both evaluation entrypoints resolve their USD and texture dependencies within these directories. The remaining `gltf/pbr.mdl` dependency is supplied by Isaac Sim (`kit/mdl/core/mdl/gltf/pbr.mdl` on the source installation). Use the Isaac Sim runtime and its MDL search paths. No rendered-image or GPU closed-loop acceptance is claimed by the file/dependency audit.

Source asset ownership and applicable HSSD asset terms remain unchanged. No policy weights, recordings, or videos are included here.
