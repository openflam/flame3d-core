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

type View =
  | { name: "list" }
  | { name: "upload" }
  | {
      name: "detail";
      dataset: { name: string; status: DatasetStatus; jobId: string | null };
    };

export default function App() {
  const [view, setView] = useState<View>({ name: "list" });

  const showList = () => setView({ name: "list" });

  const openDataset = (ds: Dataset) =>
    setView({
      name: "detail",
      dataset: { name: ds.dataset_name, status: ds.status, jobId: ds.job_id },
    });

  // After an upload, jump straight to the (processing) detail view.
  const onUploadStarted = (datasetName: string, jobId: string) =>
    setView({
      name: "detail",
      dataset: { name: datasetName, status: "processing", jobId },
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
          <UploadForm onStarted={onUploadStarted} onCancel={showList} />
        )}

        {view.name === "detail" && (
          <DatasetDetail
            name={view.dataset.name}
            initialStatus={view.dataset.status}
            jobId={view.dataset.jobId}
            onBack={showList}
          />
        )}
      </Container>
    </Box>
  );
}
