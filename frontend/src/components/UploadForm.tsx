import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import LinearProgress from "@mui/material/LinearProgress";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ConfigForm from "./ConfigForm";
import {
  fetchDefaultConfig,
  uploadAndProcess,
  type Config,
} from "../api";

interface Props {
  /** Called once the upload succeeds and the pipeline has been enqueued. */
  onStarted: (datasetName: string, jobId: string) => void;
  onCancel: () => void;
}

export default function UploadForm({ onStarted, onCancel }: Props) {
  const [defaults, setDefaults] = useState<Config | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  // Upload progress as a fraction in [0, 1]; null when not uploading.
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);

  useEffect(() => {
    fetchDefaultConfig()
      .then((cfg) => {
        setDefaults(cfg);
        setConfig(cfg);
      })
      .catch((e) => setLoadError((e as Error).message));
  }, []);

  const handleSubmit = async () => {
    if (!file || !config) return;
    setSubmitting(true);
    setSubmitError(null);
    setUploadProgress(0);
    try {
      const { job_id } = await uploadAndProcess(file, config, setUploadProgress);
      const name = String(config.dataset_name ?? "");
      onStarted(name, job_id);
    } catch (e) {
      setSubmitError((e as Error).message);
    } finally {
      setSubmitting(false);
      setUploadProgress(null);
    }
  };

  // Once all bytes are sent, the server is still saving + enqueuing — show an
  // indeterminate bar for that tail rather than sitting at a static 100%.
  const uploading = uploadProgress !== null;
  const uploadPct = Math.round((uploadProgress ?? 0) * 100);
  const serverFinishing = uploading && uploadPct >= 100;

  return (
    <Box>
      <Button startIcon={<ArrowBackIcon />} onClick={onCancel} sx={{ mb: 2 }}>
        Back to datasets
      </Button>

      <Typography variant="h5" sx={{ mb: 0.5 }}>
        New dataset
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure the pipeline, upload a raw-data export, and start processing.
      </Typography>

      {loadError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load default config: {loadError}
        </Alert>
      )}

      {config && defaults && (
        <>
          <Paper variant="outlined" sx={{ p: 2.5, mb: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
              Dataset archive
            </Typography>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Button
                component="label"
                variant="outlined"
                startIcon={<UploadFileIcon />}
              >
                Choose .zip
                <input
                  hidden
                  type="file"
                  accept=".zip,application/zip"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </Button>
              <Typography variant="body2" color="text.secondary">
                {file ? file.name : "No file selected"}
              </Typography>
            </Box>
          </Paper>

          <ConfigForm config={config} defaults={defaults} onChange={setConfig} />

          {submitError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {submitError}
            </Alert>
          )}

          <Box sx={{ display: "flex", gap: 2, mt: 3 }}>
            <Button
              variant="contained"
              size="large"
              startIcon={
                submitting ? (
                  <CircularProgress size={18} color="inherit" />
                ) : (
                  <PlayArrowIcon />
                )
              }
              disabled={!file || submitting}
              onClick={handleSubmit}
            >
              {!submitting
                ? "Start processing"
                : serverFinishing
                  ? "Starting…"
                  : `Uploading… ${uploadPct}%`}
            </Button>
            <Button
              onClick={() => defaults && setConfig(defaults)}
              disabled={submitting}
            >
              Reset defaults
            </Button>
          </Box>

          {uploading && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress
                variant={serverFinishing ? "indeterminate" : "determinate"}
                value={uploadPct}
              />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 0.5, display: "block" }}
              >
                {serverFinishing
                  ? "Upload complete — preparing pipeline…"
                  : `Uploading ${file?.name ?? ""} — ${uploadPct}%`}
              </Typography>
            </Box>
          )}
        </>
      )}
    </Box>
  );
}
