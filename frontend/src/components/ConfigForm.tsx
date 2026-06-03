import Accordion from "@mui/material/Accordion";
import AccordionDetails from "@mui/material/AccordionDetails";
import AccordionSummary from "@mui/material/AccordionSummary";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ConfigField, { kindOf } from "./ConfigField";
import type { Config, ConfigValue } from "../api";

interface Props {
  config: Config;
  defaults: Config;
  onChange: (config: Config) => void;
}

function humanize(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Immutably set a nested value addressed by `path`. */
function setAtPath(obj: ConfigValue, path: string[], value: ConfigValue): ConfigValue {
  if (path.length === 0) return value;
  const [head, ...rest] = path;
  const base = (obj ?? {}) as Record<string, ConfigValue>;
  return { ...base, [head]: setAtPath(base[head], rest, value) };
}

function isObject(v: ConfigValue): v is Record<string, ConfigValue> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Renders the full master config as a form. Top-level primitive fields go in a
 * "General" card; each nested object becomes a collapsible section. The widget
 * for every leaf is generated from the default config's value types, so the
 * whole form is data-driven — no per-field code.
 */
export default function ConfigForm({ config, defaults, onChange }: Props) {
  const set = (path: string[], value: ConfigValue) =>
    onChange(setAtPath(config, path, value) as Config);

  const topKeys = Object.keys(defaults);
  const generalKeys = topKeys.filter((k) => !isObject(defaults[k]));
  const sectionKeys = topKeys.filter((k) => isObject(defaults[k]));

  const renderLeaf = (path: string[], key: string, defVal: ConfigValue) => {
    // Walk `config` along the path to find the current value.
    let current: ConfigValue = config;
    for (const p of path) {
      current = isObject(current) ? current[p] : undefined!;
    }
    return (
      <ConfigField
        key={path.join(".")}
        fieldKey={key}
        kind={kindOf(defVal)}
        value={current}
        onChange={(v) => set(path, v)}
      />
    );
  };

  return (
    <Box>
      <Paper variant="outlined" sx={{ p: 2.5, mb: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 2 }}>
          General
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
            gap: 2,
          }}
        >
          {generalKeys.map((k) => renderLeaf([k], k, defaults[k]))}
        </Box>
      </Paper>

      {sectionKeys.map((section) => {
        const sectionDefaults = defaults[section] as Record<string, ConfigValue>;
        const keys = Object.keys(sectionDefaults);
        return (
          <Accordion
            key={section}
            disableGutters
            variant="outlined"
            sx={{ "&:before": { display: "none" }, mb: 1 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2">{humanize(section)}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
                  gap: 2,
                  alignItems: "center",
                }}
              >
                {keys.map((k) =>
                  renderLeaf([section, k], k, sectionDefaults[k]),
                )}
              </Box>
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Box>
  );
}
