import { createTheme } from "@mui/material/styles";

// A clean, professional light theme. Neutral greys with a single restrained
// accent so the UI stays minimal.
export const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#2563eb" },
    background: {
      default: "#f7f8fa",
      paper: "#ffffff",
    },
    text: {
      primary: "#1f2937",
      secondary: "#6b7280",
    },
  },
  shape: { borderRadius: 8 },
  typography: {
    fontFamily:
      '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
    subtitle2: { fontWeight: 600 },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: "none" },
      },
    },
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: { root: { textTransform: "none", fontWeight: 600 } },
    },
  },
});
