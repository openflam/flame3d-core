import { useEffect, useRef, useState } from "react";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Container from "@mui/material/Container";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import ConfigForm from "./components/ConfigForm";
import PipelineProgress from "./components/PipelineProgress";
import {
  fetchDefaultConfig,
  fetchJob,
  uploadAndProcess,
  type Config,
  type Job,
} from "./api";

export default function App() {
  const [defaults, setDefaults] = useState<Config | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const pollRef = useRef<number | null>(null);

  // Load default config on mount.
  useEffect(() => {
    fetchDefaultConfig()
      .then((cfg) => {
        setDefaults(cfg);
        setConfig(cfg);
      })
      .catch((e) => setLoadError(e.message));
  }, []);

  // Poll the active job until it finishes.
  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    if (pollRef.current) return; // already polling
    pollRef.current = window.setInterval(async () => {
      try {
        const updated = await fetchJob(job.id);
        setJob(updated);
      } catch {
        /* transient errors are ignored; next tick retries */
      }
    }, 1500);
    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [job]);

  const handleSubmit = async () => {
    if (!file || !config) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const { job_id } = await uploadAndProcess(file, config);
      const initial = await fetchJob(job_id);
      setJob(initial);
    } catch (e) {
      setSubmitError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleReset = () => {
    setJob(null);
    setFile(null);
    setSubmitError(null);
    if (defaults) setConfig(defaults);
  };

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "background.default" }}>
      <AppBar
        position="sticky"
        elevation={0}
        color="inherit"
        sx={{ borderBottom: 1, borderColor: "divider" }}
      >
        <Toolbar>
          <Typography variant="h6" sx={{ fontWeight: 700, flexGrow: 1 }}>
            flame3d<span style={{ color: "#2563eb" }}>·core</span>
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Data Processing
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ py: 4 }}>
        {loadError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            Failed to load default config: {loadError}
          </Alert>
        )}

        {job ? (
          <Box>
            <PipelineProgress job={job} />
            <Button
              startIcon={<RestartAltIcon />}
              onClick={handleReset}
              sx={{ mt: 2 }}
            >
              New run
            </Button>
          </Box>
        ) : (
          config && defaults && (
            <Box>
              <Typography variant="h5" sx={{ mb: 0.5 }}>
                Upload &amp; Process
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Configure the pipeline, upload a raw-data export, and start
                processing.
              </Typography>

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

              <ConfigForm
                config={config}
                defaults={defaults}
                onChange={setConfig}
              />

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
                  {submitting ? "Uploading…" : "Start processing"}
                </Button>
                <Button onClick={() => defaults && setConfig(defaults)}>
                  Reset defaults
                </Button>
              </Box>
            </Box>
          )
        )}
      </Container>
    </Box>
  );
}
