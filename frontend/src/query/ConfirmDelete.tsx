import { useState } from "react";
import type { CSSProperties } from "react";

interface ConfirmDeleteProps {
  onDelete: () => void;
}

const confirmTextStyle: CSSProperties = {
  margin: 0,
  fontSize: "13px",
  color: "#374151",
  textAlign: "center",
};

const cancelButtonStyle: CSSProperties = {
  flex: 1,
  padding: "8px",
  cursor: "pointer",
  backgroundColor: "#f3f4f6",
  color: "#374151",
  border: "1px solid #d1d5db",
  borderRadius: "6px",
  fontSize: "13px",
  fontWeight: "600",
};

const confirmButtonStyle: CSSProperties = {
  flex: 1,
  padding: "8px",
  cursor: "pointer",
  backgroundColor: "#ef4444",
  color: "white",
  border: "none",
  borderRadius: "6px",
  fontSize: "13px",
  fontWeight: "600",
};

export default function ConfirmDelete({ onDelete }: ConfirmDeleteProps) {
  const [confirming, setConfirming] = useState(false);

  if (!confirming) {
    return (
      <button
        onClick={() => setConfirming(true)}
        style={{
          width: "100%",
          padding: "8px",
          cursor: "pointer",
          backgroundColor: "#fef2f2",
          color: "#ef4444",
          border: "1px solid #fecaca",
          borderRadius: "6px",
          fontSize: "13px",
          fontWeight: "600",
        }}
      >
        Delete Component
      </button>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <p style={confirmTextStyle}>
        Are you sure you want to delete this component? This cannot be undone.
      </p>
      <div style={{ display: "flex", gap: "8px" }}>
        <button onClick={() => setConfirming(false)} style={cancelButtonStyle}>
          Cancel
        </button>
        <button
          onClick={() => {
            setConfirming(false);
            onDelete();
          }}
          style={confirmButtonStyle}
        >
          Yes, Delete
        </button>
      </div>
    </div>
  );
}
