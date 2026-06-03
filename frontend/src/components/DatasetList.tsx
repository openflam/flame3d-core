import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import AddIcon from "@mui/icons-material/Add";
import RefreshIcon from "@mui/icons-material/Refresh";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { fetchDatasets, type Dataset, type DatasetStatus } from "../api";

const STATUS_CHIP: Record<
  DatasetStatus,
  { label: string; color: "success" | "warning" | "error" }
> = {
  complete: { label: "Complete", color: "success" },
  processing: { label: "Processing", color: "warning" },
  failed: { label: "Failed", color: "error" },
};

interface Props {
  onSelect: (dataset: Dataset) => void;
  onUploadNew: () => void;
}

export default function DatasetList({ onSelect, onUploadNew }: Props) {
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setError(null);
    fetchDatasets()
      .then(setDatasets)
      .catch((e) => setError((e as Error).message));
  };

  useEffect(load, []);

  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 2,
        }}
      >
        <Box>
          <Typography variant="h5">Datasets</Typography>
          <Typography variant="body2" color="text.secondary">
            Select a dataset to view its processing status.
          </Typography>
        </Box>
        <Box sx={{ display: "flex", gap: 1 }}>
          <IconButton onClick={load} aria-label="refresh">
            <RefreshIcon />
          </IconButton>
          <Button variant="contained" startIcon={<AddIcon />} onClick={onUploadNew}>
            New dataset
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {datasets === null && !error ? (
        <Box sx={{ display: "flex", justifyContent: "center", py: 6 }}>
          <CircularProgress />
        </Box>
      ) : datasets && datasets.length === 0 ? (
        <Paper variant="outlined" sx={{ p: 4, textAlign: "center" }}>
          <Typography color="text.secondary">
            No datasets yet. Upload one to get started.
          </Typography>
        </Paper>
      ) : (
        <Paper variant="outlined">
          <List disablePadding>
            {datasets?.map((ds, i) => {
              const chip = STATUS_CHIP[ds.status] ?? STATUS_CHIP.processing;
              return (
                <ListItemButton
                  key={ds.dataset_name}
                  onClick={() => onSelect(ds)}
                  divider={i < datasets.length - 1}
                  sx={{ py: 1.5 }}
                >
                  <ListItemText
                    primary={ds.dataset_name}
                    secondary={ds.data_source ?? undefined}
                    primaryTypographyProps={{ fontWeight: 600 }}
                  />
                  <Chip
                    size="small"
                    label={chip.label}
                    color={chip.color}
                    variant="outlined"
                    sx={{ mr: 1 }}
                  />
                  <ChevronRightIcon color="disabled" />
                </ListItemButton>
              );
            })}
          </List>
        </Paper>
      )}
    </Box>
  );
}
