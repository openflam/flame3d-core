import { useEffect, useState } from "react";
import FormControlLabel from "@mui/material/FormControlLabel";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import type { ConfigValue } from "../api";

export type FieldKind = "bool" | "number" | "string" | "null" | "array";

/** Derive a stable field kind from the *default* value's type. */
export function kindOf(value: ConfigValue): FieldKind {
  if (typeof value === "boolean") return "bool";
  if (typeof value === "number") return "number";
  if (typeof value === "string") return "string";
  if (Array.isArray(value)) return "array";
  return "null";
}

function humanize(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

interface Props {
  fieldKey: string;
  kind: FieldKind;
  value: ConfigValue;
  onChange: (value: ConfigValue) => void;
}

/**
 * Renders a single editable config leaf. The widget is chosen from the field's
 * `kind` (derived once from the default config), so e.g. a field that defaults
 * to `null` keeps null-on-empty semantics even after the user types into it.
 */
export default function ConfigField({ fieldKey, kind, value, onChange }: Props) {
  const label = humanize(fieldKey);

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

  // Text-like fields (number / string / null / array) use a local string
  // buffer so the user can type freely; the typed value is converted to the
  // appropriate JSON type on every change.
  const toBuffer = (v: ConfigValue): string => {
    if (v === null || v === undefined) return "";
    if (Array.isArray(v)) return v.join(", ");
    return String(v);
  };

  const [buffer, setBuffer] = useState<string>(toBuffer(value));

  // Keep the buffer in sync if the value is reset externally (e.g. "Reset
  // defaults"). Avoid clobbering while the user is mid-edit on this field by
  // only syncing when the canonical value differs from what we'd emit.
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
          ? []
          : trimmed.split(",").map((s) => s.trim()).filter(Boolean),
      );
      return;
    }
    if (trimmed === "") {
      // Empty number/null fields collapse to null; empty strings stay "".
      onChange(kind === "string" ? "" : null);
      return;
    }
    if (kind === "number") {
      const n = Number(trimmed);
      onChange(Number.isNaN(n) ? value : n);
      return;
    }
    if (kind === "null") {
      const n = Number(trimmed);
      onChange(trimmed !== "" && !Number.isNaN(n) ? n : raw);
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
      placeholder={kind === "null" || kind === "array" ? "null" : undefined}
      helperText={kind === "array" ? "comma-separated" : undefined}
    />
  );
}
