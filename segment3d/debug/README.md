# Debug Scripts

Standalone inspection tools for the `segment3d` pipeline outputs.

These run inside the **`segmentation-worker`** container (it has the full ML
stack and read access to `data/` and `outputs/`). Invoke each script as a
module from the project root, e.g.:

```bash
docker compose exec segmentation-worker \
    conda run --no-capture-output -n flame3d-core \
    python -m segment3d.debug.component --dataset_name <name> --component_id <int>
```

The command blocks below show just the `python -m …` part; prefix them with the
`docker compose exec … conda run …` wrapper above. They read paths via
`config_io`, so outputs land under `outputs/<dataset_name>/` as usual.

---

## `component.py` — Visualize a single connected component

Renders every mask that contributed to a given connected component and, optionally, exports its 3D point cloud.

```bash
python -m segment3d.debug.component --dataset_name <name> --component_id <int> [--display] [--only-mask]
```

| Flag             | Default      | Description                                                |
| ---------------- | ------------ | ---------------------------------------------------------- |
| `--dataset_name` | _(required)_ | Dataset folder name under `data/` and `outputs/`           |
| `--component_id` | _(required)_ | `connected_comp_id` to inspect                             |
| `--display`      | off          | Show rendered images interactively via matplotlib          |
| `--only-mask`    | off          | Crop to masked pixels only (white background outside mask) |

**Output** — written to `outputs/<dataset_name>/debug_components/component_<id>/`:

- One PNG per contributing mask showing the mask as a semi-transparent orange overlay on the source frame, with the mask ID labelled in the corner.
- `component_<id>.ply` — binary PLY point cloud of the full scene, with the component's 3D points highlighted in orange and all other points in grey. Open in MeshLab, CloudCompare, etc.

Supports both connected-component formats:

- `mask_id_set` (e.g. `frame_00001_mask_5`) — reads SAM masks from the masks directory.
- `instance_ids` (e.g. `Boxes_seq_0_0`) — reads per-frame JSONs from `object_level_masks/masks/`.

---

## `instance.py` — Visualize a single object instance in 3D

Exports a PLY point cloud of the full scene with a specific object instance's 3D points highlighted in red and all other points in white.

```bash
python -m segment3d.debug.instance --dataset_name <name> --instance_name <instance>
```

| Flag              | Default      | Description                                                                              |
| ----------------- | ------------ | ---------------------------------------------------------------------------------------- |
| `--dataset_name`  | _(required)_ | Dataset folder name under `data/` and `outputs/`                                         |
| `--instance_name` | _(required)_ | Instance in the format `{object_name}_seq_{seq_id}_{object_id}` (e.g. `battery_seq_0_5`) |

**Output** — `outputs/<dataset_name>/debug_component/instance_<instance_name>.ply`:

- Binary PLY point cloud of the full scene.
- Instance points coloured **red**; all other scene points **white**.
- Open in MeshLab, CloudCompare, etc.

Reads `outputs/<dataset_name>/object_level_masks/object_3d_associations.json` to resolve the instance's 3D point IDs.

---

## `inter_components.py` — Compare instances across two connected components

Inspects cross-component instance overlap between two connected components.
By default it is fast: it reads stored values directly from `unmerged_edges.json`.
Pass `--fresh-calculate` to recompute everything from scratch (loads COLMAP + CLIP).

```bash
# Fast path – read from unmerged_edges.json only
python -m segment3d.debug.inter_components --dataset_name <name> --comp_id_a <int> --comp_id_b <int>

# Full recalculation – loads COLMAP model and CLIP model
python -m segment3d.debug.inter_components --dataset_name <name> --comp_id_a <int> --comp_id_b <int> --fresh-calculate
```

| Flag                | Default                   | Description                                                          |
| ------------------- | ------------------------- | -------------------------------------------------------------------- |
| `--dataset_name`    | _(required)_              | Dataset folder name under `data/` and `outputs/`                     |
| `--comp_id_a`       | _(required)_              | First connected-component ID                                         |
| `--comp_id_b`       | _(required)_              | Second connected-component ID                                        |
| `--voxel_size_cm`   | from `DEFAULT_PARAMETERS` | Voxel side length used for the Jaccard grid                          |
| `--fresh-calculate` | off                       | Recompute voxel Jaccard, containment, and CLIP distance from scratch |

**Output** — printed to stdout in two sections:

1. **Calculated values** _(only with `--fresh-calculate`)_ — a table over every cross-component instance pair with columns:
   - `Jaccard` — voxel-grid intersection-over-union of the two instances' 3D point sets.
   - `Contain` — intersection / min(|A|, |B|), i.e. how much the smaller instance is contained in the larger.
   - `CLIPdist` — cosine distance (1 − dot product) between the two instances' CLIP image embeddings.

2. **Unmerged edges from file** — the matching rows from `unmerged_edges.json` for these component pairs, showing the stored Jaccard, CLIP distance, and the rejection reason that prevented the merge.

---

## `inter_instance.py` — Trace a path between two instances within a component

Finds and prints the shortest hop-count path between two instance IDs inside a
single connected component, using the `edges` stored in `connected_components.json`.
Each hop is annotated with its Jaccard similarity and CLIP distance.

```bash
python -m segment3d.debug.inter_instance \
    --dataset_name <name> \
    --component_id <int> \
    --source_inst_id <instance_id> \
    --dest_inst_id <instance_id>
```

| Flag               | Default      | Description                                      |
| ------------------ | ------------ | ------------------------------------------------ |
| `--dataset_name`   | _(required)_ | Dataset folder name under `data/` and `outputs/` |
| `--component_id`   | _(required)_ | `connected_comp_id` whose edge graph to search   |
| `--source_inst_id` | _(required)_ | Starting instance ID (e.g. `battery_seq_0_1`)    |
| `--dest_inst_id`   | _(required)_ | Destination instance ID (e.g. `battery_seq_0_2`) |

**Output** — printed to stdout:

- The path length (number of hops).
- Each node in the path, with `▶` marking the source and `★` marking the destination.
- Between each pair of nodes: the edge's `jaccard` and `clip_distance` values.
- A warning if either instance ID is not listed in the component's `instance_ids`.
- A clear message if no path exists (the two instances are in disconnected sub-graphs within the component).
