import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import Paper from "@mui/material/Paper";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import AddIcon from "@mui/icons-material/Add";
import RefreshIcon from "@mui/icons-material/Refresh";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ViewInArIcon from "@mui/icons-material/ViewInAr";
import { deleteDataset, fetchDatasets, type Dataset, type DatasetStatus } from "../api";

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
  onQuery: (dataset: Dataset) => void;
}

export default function DatasetList({ onSelect, onUploadNew, onQuery }: Props) {
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Dataset pending delete-confirmation, plus in-flight / error state.
  const [toDelete, setToDelete] = useState<Dataset | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const load = () => {
    setError(null);
    fetchDatasets()
      .then(setDatasets)
      .catch((e) => setError((e as Error).message));
  };

  useEffect(load, []);

  const confirmDelete = async () => {
    if (!toDelete) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteDataset(toDelete.dataset_name);
      setToDelete(null);
      load();
    } catch (e) {
      setDeleteError((e as Error).message);
    } finally {
      setDeleting(false);
    }
  };

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
                  {ds.status === "complete" && (
                    <Button
                      size="small"
                      variant="outlined"
                      startIcon={<ViewInArIcon fontSize="small" />}
                      onClick={(e) => {
                        // Don't trigger the row's onSelect navigation.
                        e.stopPropagation();
                        onQuery(ds);
                      }}
                      sx={{ mr: 1 }}
                    >
                      Query
                    </Button>
                  )}
                  <Tooltip title="Delete dataset">
                    <IconButton
                      edge="end"
                      aria-label={`delete ${ds.dataset_name}`}
                      onClick={(e) => {
                        // Don't trigger the row's onSelect navigation.
                        e.stopPropagation();
                        setDeleteError(null);
                        setToDelete(ds);
                      }}
                      sx={{ mr: 0.5 }}
                    >
                      <DeleteOutlineIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <ChevronRightIcon color="disabled" />
                </ListItemButton>
              );
            })}
          </List>
        </Paper>
      )}

      <Dialog open={toDelete !== null} onClose={() => !deleting && setToDelete(null)}>
        <DialogTitle>Delete dataset?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            <strong>{toDelete?.dataset_name}</strong> will be hidden from the
            list. Its files on disk are removed separately by the{" "}
            <code>clean_storage</code> cleanup job.
          </DialogContentText>
          {deleteError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {deleteError}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setToDelete(null)} disabled={deleting}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={confirmDelete}
            disabled={deleting}
            startIcon={
              deleting ? <CircularProgress size={16} color="inherit" /> : undefined
            }
          >
            {deleting ? "Deleting…" : "Delete"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
