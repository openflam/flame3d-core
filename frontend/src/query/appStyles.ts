import type { CSSProperties } from "react";

export const styles = {
  rootContainer: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    width: "100vw",
    overflow: "hidden",
  } satisfies CSSProperties,

  topBarArea: {
    padding: "8px 16px 0",
    borderBottom: "1px solid #dee2e6",
    backgroundColor: "#fff",
    zIndex: 1,
    flexShrink: 0,
  } satisfies CSSProperties,

  mainContentArea: {
    display: "flex",
    flex: 1,
    minHeight: 0,
  } satisfies CSSProperties,

  leftColumnBase: {
    overflow: "hidden",
    transition: "width 0.3s ease",
    display: "flex",
    flexDirection: "row",
    backgroundColor: "#f8f9fa",
    flexShrink: 0,
  } satisfies CSSProperties,

  searchResultBox: {
    flex: 1,
    padding: "16px",
    overflowY: "auto",
    minWidth: 0,
  } satisfies CSSProperties,

  searchComponentListWrapper: {
    width: "160px",
    flexShrink: 0,
    overflow: "hidden",
    display: "flex",
    justifyContent: "center",
    borderLeft: "1px solid #dee2e6",
  } satisfies CSSProperties,

  collapseToggleButton: {
    width: "20px",
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
    zIndex: 2,
  } satisfies CSSProperties,

  viewerContainer: {
    flex: 1,
    minWidth: 0,
    height: "100%",
    position: "relative",
  } satisfies CSSProperties,
};
