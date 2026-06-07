// Thin API client for the Flask backend. All requests go through the Vite
// dev-server proxy at `/api`, so no base URL / CORS handling is needed.

export type ConfigValue =
  | string
  | number
  | boolean
  | null
  | ConfigValue[]
  | { [key: string]: ConfigValue };

export type Config = { [key: string]: ConfigValue };

// ── Config schema (from server/config_schema.json) ─────────────────────────
// Describes how to render each config field. A leaf has a `type`; anything
// without one is a section whose keys are nested fields.

export type SchemaType = "string" | "number" | "integer" | "boolean" | "array";

export interface ConfigSchemaLeaf {
  type: SchemaType;
  possible_values?: ConfigValue[];
  nullable?: boolean;
}

export type ConfigSchemaNode =
  | ConfigSchemaLeaf
  | { [key: string]: ConfigSchemaNode };

export type ConfigSchema = { [key: string]: ConfigSchemaNode };

/** A schema node is a leaf when it carries a `type`; otherwise it's a section. */
export function isSchemaLeaf(node: ConfigSchemaNode): node is ConfigSchemaLeaf {
  return (
    typeof node === "object" &&
    node !== null &&
    typeof (node as ConfigSchemaLeaf).type === "string"
  );
}

export type StepStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "skipped";

export interface JobStep {
  name: string;
  description: string;
  status: StepStatus;
  duration_seconds: number | null;
  error: string | null;
}

export interface Job {
  id: string;
  dataset_name: string;
  data_source: string;
  status: "pending" | "running" | "completed" | "failed";
  steps: JobStep[];
  error: string | null;
}

export type DatasetStatus = "processing" | "complete" | "failed";

export interface Dataset {
  dataset_name: string;
  data_source: string | null;
  status: DatasetStatus;
  job_id: string | null;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.error ?? detail;
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function fetchDefaultConfig(): Promise<Config> {
  return asJson<Config>(await fetch("/api/config"));
}

/** The schema describing the config form's fields, types, and allowed values. */
export async function fetchConfigSchema(): Promise<ConfigSchema> {
  return asJson<ConfigSchema>(await fetch("/api/config/schema"));
}

/**
 * Upload the dataset zip + config and start processing.
 *
 * Uses XMLHttpRequest (not fetch) so we can report upload progress. `onProgress`
 * receives a fraction in [0, 1] while bytes are being sent; once the upload
 * finishes the server still needs to save + enqueue, signalled by `lengthComputable`
 * reaching 1.
 */
export function uploadAndProcess(
  file: File,
  config: Config,
  onProgress?: (fraction: number) => void,
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  form.append("config", JSON.stringify(config));

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };

    xhr.onload = () => {
      let body: { job_id?: string; error?: string } | null = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON response */
      }
      if (xhr.status >= 200 && xhr.status < 300 && body?.job_id) {
        resolve({ job_id: body.job_id });
      } else {
        reject(new Error(body?.error ?? xhr.statusText ?? "Upload failed"));
      }
    };

    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.onabort = () => reject(new Error("Upload aborted"));

    xhr.send(form);
  });
}

export async function fetchJob(jobId: string): Promise<Job> {
  return asJson<Job>(await fetch(`/api/jobs/${jobId}`));
}

export async function fetchDatasets(): Promise<Dataset[]> {
  return asJson<Dataset[]>(await fetch("/api/datasets"));
}

export async function fetchDataset(name: string): Promise<Dataset> {
  return asJson<Dataset>(await fetch(`/api/datasets/${encodeURIComponent(name)}`));
}

export interface PipelineStepInfo {
  name: string;
  description: string;
}

export async function fetchSteps(source: string): Promise<PipelineStepInfo[]> {
  return asJson<PipelineStepInfo[]>(
    await fetch(`/api/steps?source=${encodeURIComponent(source)}`),
  );
}

/** The config that was used to process a dataset (for pre-filling the form). */
export async function fetchDatasetConfig(name: string): Promise<Config> {
  return asJson<Config>(
    await fetch(`/api/datasets/${encodeURIComponent(name)}/config`),
  );
}

export interface ReprocessRequest {
  config: Config;
  as_copy: boolean;
  new_name?: string;
}

/** Re-run a dataset's pipeline, in place or as a named copy. */
export async function reprocessDataset(
  name: string,
  req: ReprocessRequest,
): Promise<{ dataset_name: string; job_id: string }> {
  return asJson<{ dataset_name: string; job_id: string }>(
    await fetch(`/api/datasets/${encodeURIComponent(name)}/reprocess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  );
}

/** Soft-delete a dataset (marks it for deletion; files purged out-of-band). */
export async function deleteDataset(name: string): Promise<void> {
  const res = await fetch(`/api/datasets/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.error ?? detail;
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new Error(detail);
  }
}
