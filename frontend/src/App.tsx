import { useState } from "react";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Container from "@mui/material/Container";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import DatasetList from "./components/DatasetList";
import DatasetDetail from "./components/DatasetDetail";
import UploadForm from "./components/UploadForm";
import type { Dataset, DatasetStatus } from "./api";

type DetailDataset = {
  name: string;
  status: DatasetStatus;
  jobId: string | null;
  dataSource: string | null;
};

type View =
  | { name: "list" }
  | { name: "upload" }
  | { name: "detail"; dataset: DetailDataset };

export default function App() {
  const [view, setView] = useState<View>({ name: "list" });

  const showList = () => setView({ name: "list" });

  const openDataset = (ds: Dataset) =>
    setView({
      name: "detail",
      dataset: {
        name: ds.dataset_name,
        status: ds.status,
        jobId: ds.job_id,
        dataSource: ds.data_source,
      },
    });

  // After an upload or re-process, jump straight to the (processing) run.
  const onProcessingStarted = (
    datasetName: string,
    jobId: string,
    dataSource: string | null = null,
  ) =>
    setView({
      name: "detail",
      dataset: { name: datasetName, status: "processing", jobId, dataSource },
    });

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "background.default" }}>
      <AppBar
        position="sticky"
        elevation={0}
        color="inherit"
        sx={{ borderBottom: 1, borderColor: "divider" }}
      >
        <Toolbar>
          <Typography
            variant="h6"
            sx={{ fontWeight: 700, flexGrow: 1, cursor: "pointer" }}
            onClick={showList}
          >
            flame3d<span style={{ color: "#2563eb" }}>·core</span>
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Data Processing
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ py: 4 }}>
        {view.name === "list" && (
          <DatasetList
            onSelect={openDataset}
            onUploadNew={() => setView({ name: "upload" })}
          />
        )}

        {view.name === "upload" && (
          <UploadForm onStarted={onProcessingStarted} onCancel={showList} />
        )}

        {view.name === "detail" && (
          <DatasetDetail
            key={`${view.dataset.name}:${view.dataset.jobId}`}
            name={view.dataset.name}
            initialStatus={view.dataset.status}
            jobId={view.dataset.jobId}
            dataSource={view.dataset.dataSource}
            onBack={showList}
            onReprocess={(datasetName, jobId) =>
              onProcessingStarted(datasetName, jobId, view.dataset.dataSource)
            }
          />
        )}
      </Container>
    </Box>
  );
}
