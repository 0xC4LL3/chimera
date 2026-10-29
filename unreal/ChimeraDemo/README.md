# Chimera engine preview

This UE 5.8 project imports the actual FBX exported by Chimera as a skeletal mesh and animation assets. A small editor script builds a studio stage with three cameras. The level also saves native single-animation playback for Play In Editor.

**Status:** source prepared. Import, material appearance, lighting and animation playback still need a live Unreal run. The configured FBX is the inherited rear take used to prepare the stage.

## Build the scene

1. Wait for the shared GPU slot described in [RESOURCE_COORDINATION.md](../../RESOURCE_COORDINATION.md).
2. Edit `Config/demo.json`. Set `source_fbx` to the chosen export, relative to the Chimera repository. `animation_hint` must select exactly one imported animation by name. The initial value selects the in-place clip.
3. Run `launch.ps1 -Build` from Windows PowerShell. The project may be on a WSL UNC path. The script uses the installed UE 5.8 by default; pass `-Engine` for a different installation path.

The build creates `/Game/Chimera/Stage/L_Chimera` and writes `Saved/demo-state.json` with the actual source file, imported assets, duration and bone count. Rebuilding replaces this generated level. The source FBX stays untouched.

The stage uses a charcoal floor, a low plinth with a copper edge and three soft lights. Camera positions are saved as `Camera / hero`, `Camera / side` and `Camera / front`. Lighting and framing are starting values for the live visual review.

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

## Storage

`Content`, `Saved`, `Intermediate` and `DerivedDataCache` are generated and ignored by Git. The UE 5.8 cache graph reads the installed engine cache and writes new derived data into this project. No C++ module or build is required.

The importer uses Epic's [FBX import options](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/FbxImportUI?application_version=5.6) and [serialized skeletal animation playback](https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/SkeletalMeshComponent?application_version=5.6). The relevant functions were also checked in the installed UE 5.8 headers. Cache settings follow the installed engine's `DerivedDataCacheGraphs` format.
