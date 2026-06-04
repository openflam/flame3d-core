import type { CSSProperties } from "react";

export const styles = {
  root: {
    display: "flex",
    flexShrink: 0,
    height: "100%",
  } satisfies CSSProperties,

  list: {
    width: "160px",
    flexShrink: 0,
    height: "100%",
    overflowY: "auto",
    overflowX: "hidden",
    background: "#fff",
    display: "flex",
    flexDirection: "column",
    gap: "6px",
    padding: "6px",
    boxSizing: "border-box",
    borderRight: "1px solid #dee2e6",
  } satisfies CSSProperties,

  card: {
    cursor: "pointer",
    borderRadius: "6px",
    overflow: "hidden",
    flexShrink: 0,
    background: "#f8f9fa",
    position: "relative",
    aspectRatio: "1 / 1",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "border-color 0.15s",
  } satisfies CSSProperties,

  cardBorderDefault: {
    border: "2px solid #dee2e6",
  } satisfies CSSProperties,

  cardBorderFocused: {
    border: "2px solid #0d6efd",
  } satisfies CSSProperties,

  image: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    display: "block",
    background: "#fff",
  } satisfies CSSProperties,

  noImage: {
    color: "#999",
    fontSize: "0.7rem",
  } satisfies CSSProperties,

  componentIdBadge: {
    position: "absolute",
    top: 3,
    left: 3,
    background: "rgba(255,255,255,0.88)",
    border: "1px solid #ced4da",
    borderRadius: "3px",
    padding: "1px 3px",
    fontSize: "0.55rem",
    lineHeight: 1.3,
    color: "#343a40",
    maxWidth: "calc(100% - 24px)",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    pointerEvents: "none",
  } satisfies CSSProperties,

  collapseButton: {
    width: "18px",
    height: "100%",
    border: "none",
    borderRight: "1px solid #dee2e6",
    background: "#e9ecef",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 0,
    flexShrink: 0,
    color: "#495057",
    fontSize: "0.6rem",
    userSelect: "none",
  } satisfies CSSProperties,
};
