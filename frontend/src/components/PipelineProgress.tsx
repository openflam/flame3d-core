import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import Paper from "@mui/material/Paper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Stepper from "@mui/material/Stepper";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import RemoveCircleOutlineIcon from "@mui/icons-material/RemoveCircleOutline";
import type { Job, JobStep } from "../api";

function StepIcon({ status }: { status: JobStep["status"] }) {
  switch (status) {
    case "completed":
      return <CheckCircleIcon color="success" />;
    case "failed":
      return <ErrorIcon color="error" />;
    case "running":
      return <CircularProgress size={22} thickness={5} />;
    case "skipped":
      return <RemoveCircleOutlineIcon color="disabled" />;
    default:
      return <RadioButtonUncheckedIcon color="disabled" />;
  }
}

function humanize(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const STATUS_LABEL: Record<Job["status"], string> = {
  pending: "Queued",
  running: "Processing…",
  completed: "Completed",
  failed: "Failed",
};

interface Props {
  job: Job;
}

export default function PipelineProgress({ job }: Props) {
  const activeStep = job.steps.findIndex((s) => s.status === "running");

  return (
    <Paper variant="outlined" sx={{ p: 3 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1 }}>
        <Typography variant="h6">{job.dataset_name}</Typography>
        <Typography
          variant="body2"
          sx={{
            px: 1,
            py: 0.25,
            borderRadius: 1,
            bgcolor:
              job.status === "failed"
                ? "#fef2f2"
                : job.status === "completed"
                  ? "#ecfdf5"
                  : "action.hover",
            color:
              job.status === "failed"
                ? "error.main"
                : job.status === "completed"
                  ? "success.main"
                  : "text.secondary",
            fontWeight: 600,
          }}
        >
          {STATUS_LABEL[job.status]}
        </Typography>
      </Box>

      <Stepper activeStep={activeStep} orientation="vertical" sx={{ mt: 2 }}>
        {job.steps.map((step) => (
          <Step key={step.name} active expanded completed={step.status === "completed"}>
            <StepLabel
              StepIconComponent={() => <StepIcon status={step.status} />}
              optional={
                <Typography variant="caption" color="text.secondary">
                  {step.description}
                  {step.duration_seconds != null &&
                    ` · ${step.duration_seconds.toFixed(1)}s`}
                  {step.status === "skipped" && " · skipped"}
                </Typography>
              }
            >
              <Typography
                sx={{
                  fontWeight: step.status === "running" ? 700 : 500,
                  color:
                    step.status === "skipped" ? "text.disabled" : "text.primary",
                }}
              >
                {humanize(step.name)}
              </Typography>
            </StepLabel>
          </Step>
        ))}
      </Stepper>

      {job.error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {job.error}
        </Alert>
      )}
    </Paper>
  );
}
