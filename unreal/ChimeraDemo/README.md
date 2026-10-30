# Chimera engine preview

This UE 5.8 project imports the actual FBX exported by Chimera as a skeletal mesh and animation assets. A small editor script builds a studio stage with three cameras. The level also saves native single-animation playback for Play In Editor.

**Status:** verified in UE 5.8 with textured FBX import, native 1× playback and three 1080p camera captures. The imported skeleton contains the 35 rig bones plus an FBX armature root.

## Build the scene

1. Close other GPU workloads. On the shared development machine, follow [RESOURCE_COORDINATION.md](../../RESOURCE_COORDINATION.md).
2. `Config/demo.json` already selects the bundled rear example and its in-place clip. To use another export, set `source_fbx` relative to the Chimera repository and choose one clip with `animation_hint`.
3. Run `launch.ps1 -Build` from Windows PowerShell. The project may be on a WSL UNC path. The script uses the installed UE 5.8 by default; pass `-Engine` for a different installation path.

The build creates `/Game/Chimera/Stage/L_Chimera` and writes `Saved/demo-state.json` with the actual source file, imported assets, duration and bone count. Rebuilding replaces the generated level and its `Hero` import folder. The source FBX stays untouched. WSL network paths use a temporary Windows drive mapping through `launch.cmd`.

The stage uses a charcoal floor, a low plinth with a copper edge and three soft lights. Camera positions are saved as `Camera / hero`, `Camera / side` and `Camera / front`. If the import message log covers the scene, select the `L_Chimera` tab.

## Review and film

Run `launch.ps1` after the first successful build. In the focused Unreal editor:

| Key | Action |
| --- | --- |
| F6 | Replay the selected imported animation |
| F7 | Pause |
| F8 | Return to frame zero |
| F9 | Switch camera |

These keys control the editor preview. For the film, label the shot **Unreal Engine preview**. Native Play In Editor uses the animation saved on the skeletal mesh actor. Check the side and front views for deformation before choosing the cinematic camera.

The recording operator can place one command in `Saved/command.py`, such as `play()`, `play(1)` to select another imported clip or `seek(1.5)`. The controller consumes that file on the next editor tick and writes the outcome to `Saved/command-result.txt`. `stop()` removes the controller callback. The keyboard controller is inactive during Play In Editor. Close Unreal to release its GPU memory.

For native playback evidence, write `import playback; playback.start()` to `Saved/command.py`. It starts Play In Editor and plays the imported clip at normal speed from each camera. It records the engine animation time, bone positions and a screenshot for each view under `Saved/playback`. The session ends automatically. Inspect this evidence to confirm playback before filming.

For clean film footage, write `import capture; capture.start(component, animations[clip_index])` to `Saved/command.py`. It renders the imported animation at 24 fps from all three cameras into `Saved/captures`. These are sampled engine preview frames. `Saved/latest-capture.txt` points to the current folder and its `status.json` reports progress. Review the first images before letting the full sequence finish. `import capture; capture.stop()` cancels it.

After capture completes, run `python3 unreal/ChimeraDemo/Scripts/encode_capture.py <capture-folder>` from WSL. It writes the three 1080p videos and their source record to `runs/submission/unreal/<capture-folder-name>`. The PNG originals stay in the project. Keep a screenshot of the real editor alongside the clean footage to show the actual import and preview.

`capture-editor.ps1 -EditorPid <pid>` records the real editor window at 1920×1080 into `Saved/editor-proof.png`. Wait for screenshot notifications to clear before using it.

## Storage

`Content`, `Saved`, `Intermediate` and `DerivedDataCache` are generated and ignored by Git. The UE 5.8 cache graph reads the installed engine cache and writes new derived data into this project. No C++ module or build is required.

The FBX contains its three JPEG textures. Unreal extracts them into the ignored `examples/chimera-rear.fbm` cache during import, so a fresh clone can recreate that folder from the FBX.

The importer uses Epic's [FBX import options](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/FbxImportUI?application_version=5.6) and [serialized skeletal animation playback](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/SkeletalMeshComponent?application_version=5.6). The relevant functions were also checked in the installed UE 5.8 headers. Cache settings follow the installed engine's `DerivedDataCacheGraphs` format.
