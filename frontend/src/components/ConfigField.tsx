import { useEffect, useState } from "react";
import Checkbox from "@mui/material/Checkbox";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormHelperText from "@mui/material/FormHelperText";
import InputLabel from "@mui/material/InputLabel";
import ListItemText from "@mui/material/ListItemText";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import type { ConfigValue, SchemaType } from "../api";

export type FieldKind = "bool" | "number" | "string" | "array";

/** Map a schema `type` to the widget kind used to render it. */
export function kindOfType(type: SchemaType): FieldKind {
  if (type === "boolean") return "bool";
  if (type === "number" || type === "integer") return "number";
  if (type === "array") return "array";
  return "string";
}

function humanize(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

interface Props {
  fieldKey: string;
  kind: FieldKind;
  value: ConfigValue;
  onChange: (value: ConfigValue) => void;
  /** When present, render a dropdown restricted to these values. */
  possibleValues?: ConfigValue[];
  /** When true, an empty/cleared field emits `null` rather than "" or staying. */
  nullable?: boolean;
}

const NULL_OPTION = "__null__";

/**
 * Renders a single editable config leaf. The widget is chosen from the field's
 * schema: a dropdown when `possibleValues` is given, a switch for booleans, and
 * a free-text field otherwise (parsed to the appropriate JSON type on change).
 */
export default function ConfigField({
  fieldKey,
  kind,
  value,
  onChange,
  possibleValues,
  nullable,
}: Props) {
  const label = humanize(fieldKey);

  // Multi-select for enumerated array fields (e.g. steps_to_run).
  if (possibleValues && possibleValues.length > 0 && kind === "array") {
    const selected: string[] = Array.isArray(value) ? value.map(String) : [];
    return (
      <FormControl size="small" fullWidth>
        <InputLabel id={`${fieldKey}-label`}>{label}</InputLabel>
        <Select
          labelId={`${fieldKey}-label`}
          label={label}
          multiple
          value={selected}
          onChange={(e) => {
            const v = e.target.value;
            const arr = typeof v === "string" ? v.split(",") : v;
            // An empty selection emits null when nullable (i.e. "use default").
            onChange(arr.length === 0 && nullable ? null : arr);
          }}
          renderValue={(sel) =>
            (sel as string[]).length === 0 ? "All" : (sel as string[]).join(", ")
          }
        >
          {possibleValues.map((opt) => (
            <MenuItem key={String(opt)} value={String(opt)}>
              <Checkbox checked={selected.indexOf(String(opt)) > -1} />
              <ListItemText primary={String(opt)} />
            </MenuItem>
          ))}
        </Select>
        <FormHelperText>Leave empty to run all</FormHelperText>
      </FormControl>
    );
  }

  // Dropdown for enumerated scalar fields.
  if (possibleValues && possibleValues.length > 0) {
    const selected = value === null || value === undefined ? NULL_OPTION : String(value);
    return (
      <FormControl size="small" fullWidth>
        <InputLabel id={`${fieldKey}-label`}>{label}</InputLabel>
        <Select
          labelId={`${fieldKey}-label`}
          label={label}
          value={selected}
          onChange={(e) => {
            const v = e.target.value;
            onChange(v === NULL_OPTION ? null : v);
          }}
        >
          {nullable && (
            <MenuItem value={NULL_OPTION}>
              <em>none</em>
            </MenuItem>
          )}
          {possibleValues.map((opt) => (
            <MenuItem key={String(opt)} value={String(opt)}>
              {String(opt)}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    );
  }

  if (kind === "bool") {
    return (
      <FormControlLabel
        control={
          <Switch
            checked={Boolean(value)}
            onChange={(e) => onChange(e.target.checked)}
          />
        }
        label={label}
        sx={{ ".MuiFormControlLabel-label": { fontSize: 14 } }}
      />
    );
  }

  // Text-like fields (number / string / array) use a local string buffer so the
  // user can type freely; the typed value is converted to JSON on every change.
  const toBuffer = (v: ConfigValue): string => {
    if (v === null || v === undefined) return "";
    if (Array.isArray(v)) return v.join(", ");
    return String(v);
  };

  // Parse a raw text buffer into the JSON value its contents represent.
  const parse = (raw: string): ConfigValue => {
    const trimmed = raw.trim();
    if (kind === "array") {
      return trimmed === ""
        ? nullable
          ? null
          : []
        : trimmed.split(",").map((s) => s.trim()).filter(Boolean);
    }
    if (trimmed === "") {
      // Empty fields collapse to null when nullable; otherwise to "" (strings)
      // or null (numbers, which have no sensible empty value).
      return nullable || kind === "number" ? null : "";
    }
    if (kind === "number") {
      const n = Number(trimmed);
      return Number.isNaN(n) ? value : n;
    }
    return raw;
  };

  const [buffer, setBuffer] = useState<string>(toBuffer(value));

  // Keep the buffer in sync if the value is reset externally (e.g. "Reset
  // defaults"). Skip when the buffer already represents `value` — otherwise
  // every keystroke (which round-trips through `onChange`) would clobber
  // in-progress text such as a trailing "," or ", " in array fields.
  useEffect(() => {
    if (JSON.stringify(parse(buffer)) !== JSON.stringify(value)) {
      setBuffer(toBuffer(value));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const emit = (raw: string) => {
    setBuffer(raw);
    onChange(parse(raw));
  };

  return (
    <TextField
      label={label}
      value={buffer}
      onChange={(e) => emit(e.target.value)}
      size="small"
      fullWidth
      type={kind === "number" ? "number" : "text"}
      placeholder={nullable ? "null" : undefined}
      helperText={kind === "array" ? "comma-separated" : undefined}
    />
  );
}
