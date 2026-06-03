import { useEffect, useRef, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Alert from "@mui/material/Alert";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import PipelineProgress from "./PipelineProgress";
import CompletedView from "./CompletedView";
import { fetchJob, type DatasetStatus, type Job } from "../api";

interface Props {
  name: string;
  initialStatus: DatasetStatus;
  jobId: string | null;
  onBack: () => void;
}

/**
 * Detail view for one dataset. If the dataset is already complete it shows the
 * (placeholder) completed view; otherwise it polls the dataset's job and shows
 * the live pipeline progress, switching to the completed view once the run
 * finishes successfully.
 */
export default function DatasetDetail({ name, initialStatus, jobId, onBack }: Props) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const alreadyComplete = initialStatus === "complete";

  useEffect(() => {
    if (alreadyComplete || !jobId) return;

    let cancelled = false;
    const tick = async () => {
      try {
        const updated = await fetchJob(jobId);
        if (cancelled) return;
        setJob(updated);
        if (updated.status === "completed" || updated.status === "failed") {
          if (pollRef.current) {
            window.clearInterval(pollRef.current);
            pollRef.current = null;
          }
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    };

    tick(); // fetch immediately, then poll
    pollRef.current = window.setInterval(tick, 1500);
    return () => {
      cancelled = true;
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [alreadyComplete, jobId]);

  const isComplete = alreadyComplete || job?.status === "completed";

  const back = (
    <Button startIcon={<ArrowBackIcon />} onClick={onBack} sx={{ mb: 2 }}>
      Back to datasets
    </Button>
  );

  let body;
  if (isComplete) {
    body = <CompletedView datasetName={name} />;
  } else if (error && !job) {
    body = <Alert severity="error">{error}</Alert>;
  } else if (!jobId) {
    body = (
      <Alert severity="info">
        No job information is available for this dataset.
      </Alert>
    );
  } else if (job) {
    body = <PipelineProgress job={job} />;
  } else {
    body = (
      <Box sx={{ display: "flex", justifyContent: "center", py: 6 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {back}
      {body}
    </Box>
  );
}
