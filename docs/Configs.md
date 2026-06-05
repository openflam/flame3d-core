# Pipeline Configuration

This document describes every field in the pipeline configuration, as defined by
[`server/config_schema.json`](../server/config_schema.json). The schema drives
both the web UI's config form and validation; the shipped defaults live in
[`server/default_config.json`](../server/default_config.json).

## Intuition: tuning the number and quality of components

A "component" is a connected group of masks that the pipeline believes is a
single physical object. The count and quality of components are mostly governed
by the [`postsam3_pipeline`](#postsam3_pipeline--post-sam3-processing) section —
specifically the **mask graph** (how masks get linked into objects) and the
**component cleaning** (how each object's 3D point cloud is filtered/split).

The two levers that matter most:

- **Edges** link masks into the same object. *More edges → bigger, fewer
  components. Fewer edges → smaller, more numerous components.* Edges are
  controlled by `tau`, `voxel_size_cm`, `K`, and `clip_distance_threshold`.
- **Cleaning** removes noise and can split one component into several. Controlled
  by `component_dbscan_eps`, `component_dbscan_min_points`, and
  `split_components`.

### Too few components (distinct objects merged into one)

Symptom: two real objects (e.g. a chair and the table next to it) end up as a
single component, or everything collapses into a few large blobs.

| Try | Direction | Why |
| --- | --- | --- |
| `tau` | **increase** (e.g. `0.4 → 0.55`) | Requires stronger 3D overlap before two masks are linked, so loosely-touching objects stay separate. |
| `voxel_size_cm` | **decrease** (e.g. `50 → 25`) | Finer voxels reduce accidental overlap between nearby-but-distinct objects. |
| `clip_distance_threshold` | **decrease** (stricter) | Forces linked masks to look visually similar, breaking links between different-looking objects. |
| `split_components` | set **`true`** | Lets component cleaning break a merged cloud into its separate DBSCAN clusters. |
| `component_dbscan_eps` | **decrease** | Tighter clustering separates two object point clouds that sit close together. |

### Too many components / over-fragmentation (one object split into pieces)

Symptom: a single object (e.g. a sofa) shows up as several components, or you
get far more components than there are objects.

| Try | Direction | Why |
| --- | --- | --- |
| `tau` | **decrease** (e.g. `0.4 → 0.25`) | Links masks with weaker overlap, joining fragments of the same object. |
| `voxel_size_cm` | **increase** (e.g. `50 → 75`) | Coarser voxels increase overlap, helping parts of one object connect. |
| `clip_distance_threshold` | **increase** (looser) | Stops visually-similar parts of one object from being disconnected. |
| `component_dbscan_eps` | **increase** | Keeps a single object's point cloud together instead of splitting it. |
| `split_components` | set **`false`** | Stops cleaning from breaking a component into multiple clusters (only noise is removed). |

### Noisy or spurious tiny components

Symptom: lots of small junk components made of stray points.

- Raise `component_dbscan_min_points` (e.g. `20 → 40`) to discard small clusters.
- Raise `min_points_in_3d_segment` so thin components aren't reported.
- Raise `segment_dbscan_min_samples` to filter per-frame mask noise before votes
  are accumulated.

### Expected objects are missing entirely

If an object never appears as a component, the problem is usually *upstream* of
the graph:

- Check `discard_objects_list` — the label may be filtered out before
  association (it discards `wall`, `floor`, `ceiling`, etc. by default).
- In [`normalize_labels`](#normalize_labels--clip-label-normalization), a high
  `min_sequence_length` drops objects seen in only a few frames before SAM3 ever
  runs. Lower it to keep short-lived objects. Enabling `fill_holes` (with a
  larger `max_gap`) also helps bridge gaps in an object's frame sequence.
- In [`identify_objects`](#identify_objects--vlm-object-identification), a small
  `max_frames` (or a weaker `model`) may simply never spot the object.

### Cleaning vs. graph — which to reach for first

If components are roughly right but each one is *spatially noisy or split*,
adjust the **component-level** knobs (`component_dbscan_*`, `split_components`).
If the wrong masks are being **grouped together or apart** in the first place,
adjust the **mask-graph** knobs (`tau`, `voxel_size_cm`, `clip_distance_threshold`).
Change one knob at a time — these interact, and small moves in `tau` /
`voxel_size_cm` have outsized effects.

## How the config is used

- **Schema** (`config_schema.json`) describes each field's `type`, its optional
  `possible_values` (rendered as a dropdown), and whether it is `nullable`.
- **Defaults** (`default_config.json`) provide the pre-filled starting values.
- The **dataset name is passed separately** from the config (via `--dataset-name`
  on the CLI, or its own field in the UI). `default_config.json` carries a
  `dataset_name`, but every pipeline step receives the name as an explicit
  argument rather than reading it from the config.

Each pipeline step is driven entirely by this config file plus the dataset name:

```bash
python -m <step.module> --config server/default_config.json --dataset-name MyDataset
```

The schema is organized into a handful of **top-level fields** and one **section
per pipeline step**. A field's `type` is one of `string`, `number`, `integer`,
`boolean`, or `array`. A `nullable` field may be left empty (`null`).

---

## Top level

| Field | Type | Allowed values | Default | Description |
| --- | --- | --- | --- | --- |
| `data_source` | string | `polycam` | `polycam` | Vendor/format of the raw input. Selects which ordered list of pipeline steps runs. |
| `start_from_step` | string (nullable) | `polycam_process`, `identify_objects`, `normalize_labels`, `sam3_segmentation`, `postsam3_pipeline`, `captioning` | `null` | Resume the pipeline from this step, reusing earlier outputs. `null` runs from the beginning. |

> **Note:** `dataset_name` is not part of the schema — it is passed separately to
> every step. See [How the config is used](#how-the-config-is-used).

---

## `polycam` — Polycam data processing

Extracts and processes a Polycam raw-data export and writes COLMAP files.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `depth_tolerance` | number | `0.05` | Relative depth tolerance for per-point visibility checks during mesh reprojection. |

---

## `identify_objects` — VLM object identification

Runs a Vision-Language Model on every frame to build an objects inventory.

| Field | Type | Allowed values | Default | Description |
| --- | --- | --- | --- | --- |
| `identifier_type` | string | `vllm`, `openai` | `vllm` | Backend: `vllm` for local GPU inference, `openai` for the OpenAI API. |
| `model` | string | — | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID (or OpenAI model name) used for identification. |
| `device` | integer | — | `0` | GPU device index for local inference. |
| `max_frames` | integer (nullable) | — | `null` | Cap on the number of frames to process. `null` processes all frames. |
| `batch_size` | integer | — | `32` | Number of frames processed per batch. |

---

## `normalize_labels` — CLIP label normalization

Normalizes the raw object labels via lemmatization + CLIP-based semantic
clustering, then builds the per-object frame index.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `distance_threshold` | number | `0.12` | Cosine-distance threshold for agglomerative clustering of label embeddings. |
| `clip_model` | string | `ViT-B-32` | OpenCLIP model name used to embed labels. |
| `clip_pretrained` | string | `openai` | OpenCLIP pretrained-weights tag. |
| `device` | string (nullable) | `null` | Torch device string (e.g. `cuda`, `cpu`, `cuda:1`). `null` auto-detects. |
| `skip_lemmatize` | boolean | `false` | When `true`, cluster raw labels directly without lemmatization. |
| `fill_holes` | boolean | `true` | Fill short gaps in each object's frame sequence. |
| `max_gap` | integer | `3` | Maximum run of consecutive missing frames to fill. Only used when `fill_holes` is `true`. |
| `objects_to_frames` | boolean | `true` | Build the `objects_to_frames.json` index consumed by SAM3. |
| `min_sequence_length` | integer | `5` | Minimum consecutive-frame run length to include in `objects_to_frames.json`. |

---

## `sam3` — SAM3 segmentation

Runs the SAM3 video predictor on per-object frame sequences.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `objects_filter` | array (nullable) | `null` | If set, process only these object names (case-insensitive). `null` processes all objects. |
| `resume` | boolean | `false` | Skip `(object, sequence)` pairs whose output directory already contains `.npz` files. |
| `objects_to_frames_path` | string (nullable) | `null` | Override path to `objects_to_frames.json`. `null` uses the default location under the outputs dir. |
| `tmp_root` | string (nullable) | `null` | Directory in which temporary JPEG folders are created. `null` uses the system default. |
| `save_images` | boolean | `false` | Also save overlay JPEGs of each mask rendered on top of the original frame. |

---

## `postsam3_pipeline` — Post-SAM3 processing

Associates per-object masks with 3D points, builds the mask graph, cleans
connected components, computes bounding boxes, and crops images. The captioning
and CLIP sub-steps here are independent of the standalone `captioning` step.

### Skip flags

Each flag, when `true`, skips that sub-step and reuses its existing output.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `skip_association` | boolean | `false` | Skip 2D→3D association (reuse existing associations). |
| `skip_graph` | boolean | `false` | Skip mask-graph building (reuse existing graph). |
| `skip_clean` | boolean | `false` | Skip DBSCAN component cleaning (reuse existing `connected_components.json`). |
| `skip_bbox` | boolean | `false` | Skip 3D bounding-box computation (reuse existing `bbox_corners.json`). |
| `skip_segment_crops` | boolean | `false` | Skip image cropping (reuse existing crops). |
| `skip_caption` | boolean | `true` | Skip the in-pipeline VLM captioning sub-step. |
| `skip_clip` | boolean | `false` | Skip CLIP embedding generation. |

### Segment-level association

Applied per per-frame mask before votes are accumulated across frames.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `discard_objects_list` | array | `["wall", "walls", "floor", "ceiling", "window"]` | Object labels (case-insensitive) excluded from association entirely. |
| `segment_dbscan_eps` | number | `0.5` | DBSCAN neighbourhood radius (in reconstruction units, typically metres). |
| `segment_dbscan_min_samples` | integer | `5` | Minimum 3D points to form a DBSCAN core sample. |

### Mask graph

| Field | Type | Allowed values | Default | Description |
| --- | --- | --- | --- | --- |
| `intersection_type` | string | `geometric`, `id_based` | `geometric` | Overlap measure: `geometric` voxelises each instance's point cloud and computes Jaccard over voxel occupancy; `id_based` computes Jaccard over raw COLMAP point-ID sets. |
| `voxel_size_cm` | number | — | `50.0` | Voxel side length in centimetres (used when `intersection_type` is `geometric`; point coords assumed to be in metres). |
| `K` | integer | — | `5` | Minimum shared-point overlap for an edge. Used only when `intersection_type` is `id_based` (edge kept when shared IDs ≥ K **or** Jaccard ≥ `tau`). |
| `tau` | number | — | `0.4` | Minimum Jaccard similarity (0–1] for an edge to be kept. |
| `clip_distance_threshold` | number | — | `0.3` | Maximum cosine distance between the two nodes' CLIP crop embeddings for an edge to be kept (0–1]; smaller is stricter. |
| `save_segment_images` | boolean | — | `true` | Save each node's representative masked-crop image for inspection. |
| `min_points` | integer | — | `1` | Minimum 3D points a mask node must have to be included in the graph. |
| `min_points_in_3d_segment` | integer | — | `10` | Minimum 3D points a connected component must have to be reported. |

### Component-level cleaning

Applied to the accumulated 3D point cloud of each connected component.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `component_dbscan_eps` | number | `0.5` | DBSCAN neighbourhood radius in world units. |
| `component_dbscan_min_samples` | integer | `5` | Minimum samples per DBSCAN core point. |
| `component_dbscan_min_points` | integer | `20` | Discard components with fewer than this many inlier points after cleaning. |
| `split_components` | boolean | `true` | Split components with multiple DBSCAN clusters into separate components. When `false`, only noise is removed. |

### Bounding box

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `percentile` | number | `95.0` | Percentile used to clip outlier points before fitting the 3D box (e.g. `95.0` ignores the most extreme 5% of points). |

### Segment crops

| Field | Type | Allowed values | Default | Description |
| --- | --- | --- | --- | --- |
| `crop_type` | string | `segment`, `bbox` | `segment` | `segment` uses per-object mask crops; `bbox` projects the 3D box to 2D and crops the projected region. |
| `top_n` | integer | — | `5` | Number of top-ranked frames to crop per connected component. |
| `min_fraction` | number | — | `0.05` | Minimum fraction of a component's 3D points that must project into a frame for it to be considered for cropping. |

### In-pipeline captioning

Used only when `skip_caption` is `false`.

| Field | Type | Allowed values | Default | Description |
| --- | --- | --- | --- | --- |
| `caption_n_images` | integer | — | `1` | Number of top crop images passed to the VLM per caption. |
| `captioner_type` | string | `vllm`, `openai` | `vllm` | Captioner backend. |
| `caption_model` | string | — | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID for the VLM captioner. |
| `caption_device` | integer | — | `0` | GPU device index for captioning. |
| `caption_batch_size` | integer | — | `32` | Batch size for captioning inference. |

### CLIP embeddings

Used only when `skip_clip` is `false`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `clip_model` | string | `ViT-H-14` | OpenCLIP model name. |
| `clip_pretrained` | string | `laion2B-s32B-b79K` | Pretrained-weights identifier for the CLIP model. |
| `clip_batch_size` | integer | `32` | Batch size for CLIP embedding generation. |
| `clip_device` | integer | `0` | GPU device index for CLIP inference. |

---

## `captioning` — Standalone captioning step

The final pipeline step: generates a caption for each component using a VLM.

| Field | Type | Allowed values | Default | Description |
| --- | --- | --- | --- | --- |
| `n_images` | integer | — | `1` | Number of top images used per component. |
| `captioner_type` | string | `vllm`, `openai` | `vllm` | Captioner backend. |
| `model` | string | — | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID (or OpenAI model name) for captioning. |
| `device` | integer | — | `0` | GPU device index. |
| `max_components` | integer (nullable) | — | `null` | Cap on the number of components to caption. `null` processes all. |
| `batch_size` | integer | — | `512` | Number of components processed per batch. |
