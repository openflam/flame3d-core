import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import Switch from "@mui/material/Switch";
import FormControlLabel from "@mui/material/FormControlLabel";
import TextField from "@mui/material/TextField";
import ReplayIcon from "@mui/icons-material/Replay";
import ConfigForm from "./ConfigForm";
import {
  fetchConfigSchema,
  fetchDatasetConfig,
  reprocessDataset,
  type Config,
  type ConfigSchema,
} from "../api";

interface Props {
  datasetName: string;
  dataSource: string | null;
  /** Called with the resulting dataset name + job id once a run is enqueued. */
  onStarted: (datasetName: string, jobId: string) => void;
}

export default function ReprocessForm({ datasetName, onStarted }: Props) {
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [asCopy, setAsCopy] = useState(false);
  const [newName, setNewName] = useState(`${datasetName}-copy`);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    Promise.all([fetchConfigSchema(), fetchDatasetConfig(datasetName)])
      .then(([sch, cfg]) => {
        setSchema(sch);
        setConfig(cfg);
      })
      .catch((e) => setLoadError((e as Error).message));
  }, [datasetName]);

  const handleSubmit = async () => {
    if (!config) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      // Which steps run is controlled by the config form's "steps_to_run".
      const { dataset_name, job_id } = await reprocessDataset(datasetName, {
        config,
        as_copy: asCopy,
        new_name: asCopy ? newName.trim() : undefined,
      });
      onStarted(dataset_name, job_id);
    } catch (e) {
      setSubmitError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" sx={{ mb: 0.5 }}>
        Re-process
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Adjust parameters and continue the pipeline from any step.
      </Typography>

      {loadError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {loadError}
        </Alert>
      )}

      {config && schema && (
        <>
          <Paper variant="outlined" sx={{ p: 2.5, mb: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={asCopy}
                  onChange={(e) => setAsCopy(e.target.checked)}
                />
              }
              label="Process as a copy (keep the original unchanged)"
            />

            {asCopy && (
              <TextField
                label="New dataset name"
                size="small"
                fullWidth
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                sx={{ mt: 1.5 }}
                helperText="Files are copied into a new dataset, then processed."
              />
            )}
          </Paper>

          <ConfigForm schema={schema} config={config} onChange={setConfig} />

          {submitError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {submitError}
            </Alert>
          )}

          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              size="large"
              startIcon={
                submitting ? (
                  <CircularProgress size={18} color="inherit" />
                ) : (
                  <ReplayIcon />
                )
              }
              disabled={submitting || (asCopy && !newName.trim())}
              onClick={handleSubmit}
            >
              {submitting
                ? "Starting…"
                : asCopy
                  ? "Process copy"
                  : "Continue processing"}
            </Button>
          </Box>
        </>
      )}
    </Box>
  );
}
