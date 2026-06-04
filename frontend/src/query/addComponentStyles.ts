import type { CSSProperties } from "react";
import { styles as baseStyles } from "./componentDetailsStyles";

export const styles = {
  ...baseStyles,
  
  fieldContainer: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  } satisfies CSSProperties,

  fieldLabel: {
    fontSize: "12px",
    color: "#666",
  } satisfies CSSProperties,

  descriptionTextarea: {
    ...baseStyles.textarea,
    height: "100px",
  } satisfies CSSProperties,

  fileInput: {
    fontSize: "12px",
  } satisfies CSSProperties,

  openButton: {
    position: "absolute",
    top: "20px",
    right: "20px",
    zIndex: 20,
    padding: "8px 16px",
    backgroundColor: "#3b82f6",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontWeight: "bold",
    boxShadow: "0 2px 5px rgba(0,0,0,0.2)",
  } satisfies CSSProperties,
};
