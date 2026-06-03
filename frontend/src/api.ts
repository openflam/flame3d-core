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

export async function uploadAndProcess(
  file: File,
  config: Config,
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  form.append("config", JSON.stringify(config));
  return asJson<{ job_id: string }>(
    await fetch("/api/upload", { method: "POST", body: form }),
  );
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
