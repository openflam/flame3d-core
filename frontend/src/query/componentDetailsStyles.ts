import type { CSSProperties } from "react";

export const styles = {
  panel: {
    position: "absolute",
    top: "70px",
    right: "20px",
    width: "300px",
    maxHeight: "calc(100% - 90px)",
    overflowY: "auto",
    zIndex: 10,
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    padding: "20px",
    borderRadius: "8px",
    boxShadow: "0 10px 25px rgba(0,0,0,0.15)",
    border: "1px solid #eee",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    backdropFilter: "blur(4px)",
  } satisfies CSSProperties,

  sectionLabel: {
    margin: 0,
    fontSize: "12px",
    color: "#888",
    letterSpacing: "1px",
    textTransform: "uppercase",
  } satisfies CSSProperties,

  loadingContainer: {
    textAlign: "center",
    padding: "20px",
  } satisfies CSSProperties,

  loadingText: {
    color: "#6b7280",
  } satisfies CSSProperties,

  imageWrapper: {
    width: "100%",
    borderRadius: "6px",
    marginBottom: "8px",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  } satisfies CSSProperties,

  image: {
    maxWidth: "100%",
    maxHeight: "220px",
    width: "auto",
    height: "auto",
    display: "block",
    borderRadius: "6px",
  } satisfies CSSProperties,

  textarea: {
    width: "100%",
    height: "240px",
    padding: "10px",
    borderRadius: "6px",
    border: "2px solid #3b82f6",
    fontSize: "14px",
    outline: "none",
    resize: "none",
    overflowY: "auto",
  } satisfies CSSProperties,

  captionBox: {
    minHeight: "60px",
    maxHeight: "240px",
    overflowY: "auto",
    padding: "10px",
    borderRadius: "6px",
    border: "1px solid #e5e7eb",
  } satisfies CSSProperties,

  captionText: {
    margin: 0,
    fontSize: "16px",
    color: "#1f2937",
    lineHeight: "1.5",
  } satisfies CSSProperties,

  buttonRow: {
    display: "flex",
    gap: "10px",
  } satisfies CSSProperties,

  gizmoRow: {
    display: "flex",
    gap: "6px",
    marginBottom: "4px",
    width: "100%",
  } satisfies CSSProperties,

  gizmoLabel: {
    fontSize: "11px",
    color: "#888",
    alignSelf: "center",
    marginRight: "4px",
  } satisfies CSSProperties,

  gizmoButton: (active: boolean): CSSProperties => ({
    flex: 1,
    padding: "6px",
    cursor: "pointer",
    backgroundColor: active ? "#3b82f6" : "#f3f4f6",
    color: active ? "white" : "#374151",
    border: `1px solid ${active ? "#3b82f6" : "#d1d5db"}`,
    borderRadius: "6px",
    fontSize: "12px",
    fontWeight: "600",
    textTransform: "capitalize",
    transition: "all 0.2s",
  }),

  editButton: (isEditing: boolean): CSSProperties => ({
    flex: 1,
    padding: "10px",
    cursor: "pointer",
    backgroundColor: isEditing ? "#fee2e2" : "#f3f4f6",
    color: isEditing ? "#ef4444" : "#374151",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "14px",
    fontWeight: "600",
    transition: "all 0.2s",
  }),

  saveButton: {
    flex: 1,
    padding: "10px",
    cursor: "pointer",
    backgroundColor: "#3b82f6",
    color: "white",
    border: "none",
    borderRadius: "6px",
    fontSize: "14px",
    fontWeight: "600",
    transition: "background-color 0.2s",
  } satisfies CSSProperties,

  warningBox: {
    padding: "8px 12px",
    borderRadius: "6px",
    backgroundColor: "#fef2f2",
    border: "1px solid #fecaca",
    color: "#b91c1c",
    fontSize: "12px",
  } satisfies CSSProperties,

  deleteSection: {
    borderTop: "1px solid #e5e7eb",
    paddingTop: "12px",
  } satisfies CSSProperties,

  deleteButton: {
    width: "100%",
    padding: "8px",
    cursor: "pointer",
    backgroundColor: "#fef2f2",
    color: "#ef4444",
    border: "1px solid #fecaca",
    borderRadius: "6px",
    fontSize: "13px",
    fontWeight: "600",
  } satisfies CSSProperties,

  dismissButton: {
    background: "none",
    border: "none",
    color: "#9ca3af",
    fontSize: "12px",
    cursor: "pointer",
    alignSelf: "center",
    marginTop: "4px",
  } satisfies CSSProperties,
};
