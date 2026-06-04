import { useEffect, useRef, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Alert from "@mui/material/Alert";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import PipelineProgress from "./PipelineProgress";
import CompletedView from "./CompletedView";
import ReprocessForm from "./ReprocessForm";
import { fetchJob, type DatasetStatus, type Job } from "../api";

interface Props {
  name: string;
  initialStatus: DatasetStatus;
  jobId: string | null;
  dataSource: string | null;
  onBack: () => void;
  /** Navigate to the run started by re-processing (in place or as a copy). */
  onReprocess: (datasetName: string, jobId: string) => void;
}

/**
 * Detail view for one dataset.
 *
 *  While the run is live, it polls the job and shows the pipeline progress.
 *  Once terminal (complete / failed), it shows a summary plus a re-process
 *    form so the user can tweak parameters and continue from any step (in
 *    place or as a copy).
 */
export default function DatasetDetail({
  name,
  initialStatus,
  jobId,
  dataSource,
  onBack,
  onReprocess,
}: Props) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const alreadyComplete = initialStatus === "complete";

  useEffect(() => {
    // No job to watch for an already-complete dataset.
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

  const jobTerminal = job?.status === "completed" || job?.status === "failed";
  // Live = the dataset was opened mid-run and the job hasn't finished yet.
  const isLive = initialStatus === "processing" && !jobTerminal;
  const isComplete = alreadyComplete || job?.status === "completed";

  const back = (
    <Button startIcon={<ArrowBackIcon />} onClick={onBack} sx={{ mb: 2 }}>
      Back to datasets
    </Button>
  );

  // ── Live run: show progress only ────────────────────────────────────────
  if (isLive) {
    let body;
    if (error && !job) {
      body = <Alert severity="error">{error}</Alert>;
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

  // ── Terminal: summary + re-process form ─────────────────────────────────
  let summary = null;
  if (isComplete) {
    summary = <CompletedView datasetName={name} />;
  } else if (job) {
    // Failed (or otherwise finished without success): show the final steps.
    summary = <PipelineProgress job={job} />;
  } else if (error) {
    summary = <Alert severity="error">{error}</Alert>;
  } else if (jobId) {
    summary = (
      <Box sx={{ display: "flex", justifyContent: "center", py: 6 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {back}
      {summary}
      <ReprocessForm
        datasetName={name}
        dataSource={dataSource}
        onStarted={onReprocess}
      />
    </Box>
  );
}
