import { useEffect, useState } from "react";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import InputLabel from "@mui/material/InputLabel";
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

  // Dropdown for enumerated fields.
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

  const [buffer, setBuffer] = useState<string>(toBuffer(value));

  // Keep the buffer in sync if the value is reset externally (e.g. "Reset
  // defaults").
  useEffect(() => {
    setBuffer(toBuffer(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const emit = (raw: string) => {
    setBuffer(raw);
    const trimmed = raw.trim();
    if (kind === "array") {
      onChange(
        trimmed === ""
          ? nullable
            ? null
            : []
          : trimmed.split(",").map((s) => s.trim()).filter(Boolean),
      );
      return;
    }
    if (trimmed === "") {
      // Empty fields collapse to null when nullable; otherwise to "" (strings)
      // or null (numbers, which have no sensible empty value).
      onChange(nullable || kind === "number" ? null : "");
      return;
    }
    if (kind === "number") {
      const n = Number(trimmed);
      onChange(Number.isNaN(n) ? value : n);
      return;
    }
    onChange(raw);
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
