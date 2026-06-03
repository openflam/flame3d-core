import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";

interface Props {
  datasetName: string;
}

/**
 * Placeholder shown for a fully-processed dataset. This will later be replaced
 * by the 3D query pipeline UI.
 */
export default function CompletedView({ datasetName }: Props) {
  return (
    <Paper variant="outlined" sx={{ p: 6, textAlign: "center" }}>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 1.5,
        }}
      >
        <CheckCircleIcon color="success" sx={{ fontSize: 56 }} />
        <Typography variant="h5">Completed</Typography>
        <Typography variant="body1" color="text.secondary">
          <strong>{datasetName}</strong> has finished processing.
        </Typography>
        <Typography variant="caption" color="text.disabled" sx={{ mt: 1 }}>
          The 3D query pipeline will appear here.
        </Typography>
      </Box>
    </Paper>
  );
}
