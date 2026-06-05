import Accordion from "@mui/material/Accordion";
import AccordionDetails from "@mui/material/AccordionDetails";
import AccordionSummary from "@mui/material/AccordionSummary";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ConfigField, { kindOfType } from "./ConfigField";
import {
  isSchemaLeaf,
  type Config,
  type ConfigSchema,
  type ConfigSchemaLeaf,
  type ConfigValue,
} from "../api";

interface Props {
  /** Field metadata (types + allowed values) describing how to render. */
  schema: ConfigSchema;
  /** Current config values being edited. */
  config: Config;
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
 * Renders the config as a form driven entirely by `schema`. Top-level leaf
 * fields go in a "General" card; each nested section becomes a collapsible
 * accordion. The widget for every field comes from its schema entry (type +
 * allowed values), so the whole form is data-driven — no per-field code.
 */
export default function ConfigForm({ schema, config, onChange }: Props) {
  const set = (path: string[], value: ConfigValue) =>
    onChange(setAtPath(config, path, value) as Config);

  const topKeys = Object.keys(schema);
  const generalKeys = topKeys.filter((k) => isSchemaLeaf(schema[k]));
  const sectionKeys = topKeys.filter((k) => !isSchemaLeaf(schema[k]));

  const renderLeaf = (path: string[], key: string, leaf: ConfigSchemaLeaf) => {
    // Walk `config` along the path to find the current value.
    let current: ConfigValue = config;
    for (const p of path) {
      current = isObject(current) ? current[p] : undefined!;
    }
    return (
      <ConfigField
        key={path.join(".")}
        fieldKey={key}
        kind={kindOfType(leaf.type)}
        value={current}
        possibleValues={leaf.possible_values}
        nullable={leaf.nullable}
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
          {generalKeys.map((k) =>
            renderLeaf([k], k, schema[k] as ConfigSchemaLeaf),
          )}
        </Box>
      </Paper>

      {sectionKeys.map((section) => {
        const sectionSchema = schema[section] as ConfigSchema;
        const keys = Object.keys(sectionSchema);
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
                  renderLeaf([section, k], k, sectionSchema[k] as ConfigSchemaLeaf),
                )}
              </Box>
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Box>
  );
}
