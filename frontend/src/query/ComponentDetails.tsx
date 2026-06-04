import type { BoundingBox } from "./types/global";
import ConfirmDelete from "./ConfirmDelete";
import { styles } from "./componentDetailsStyles";

export type GizmoMode = "translate" | "scale";

interface ComponentDetailsProps {
  editedCaption: string;
  setEditedCaption: (value: string) => void;
  isEditing: boolean;
  setIsEditing: (value: boolean) => void;
  onDismiss: () => void;
  onSave: () => void;
  saveWarning?: string | null;
  componentId: string | null;
  imageBase64?: string | null;
  isLoading?: boolean;
  editedBBox?: BoundingBox | null;
  gizmoMode?: GizmoMode;
  onGizmoModeChange?: (mode: GizmoMode) => void;
  onDelete?: () => void;
}

export default function ComponentDetails({
  editedCaption,
  setEditedCaption,
  isEditing,
  setIsEditing,
  onDismiss,
  onSave,
  saveWarning,
  componentId,
  imageBase64,
  isLoading,
  editedBBox: _editedBBox,
  gizmoMode,
  onGizmoModeChange,
  onDelete,
}: ComponentDetailsProps) {
  return (
    <div style={styles.panel}>
      <h3 style={styles.sectionLabel}>
        {componentId?.startsWith("custom_bbox_") 
          ? `Custom Location ${parseInt(componentId.split("_").pop() || "0") + 1}`
          : `Component ${componentId}`}
      </h3>

      {isLoading ? (
        <div style={styles.loadingContainer}>
          <p style={styles.loadingText}>Loading...</p>
        </div>
      ) : (
        <>
          {imageBase64 && (
            <div style={styles.imageWrapper}>
              <img
                src={`data:image/jpeg;base64,${imageBase64}`}
                alt="Component view"
                style={styles.image}
              />
            </div>
          )}

          {isEditing ? (
            <textarea
              value={editedCaption}
              onChange={(e) => setEditedCaption(e.target.value)}
              style={styles.textarea}
              autoFocus
            />
          ) : (
            <div style={styles.captionBox}>
              <p style={styles.captionText}>
                {editedCaption || "No caption available."}
              </p>
            </div>
          )}
        </>
      )}

      {isEditing && onGizmoModeChange && (
        <div style={styles.buttonRow}>
          <div style={styles.gizmoRow}>
            <span style={styles.gizmoLabel}>Gizmo:</span>
            {(["translate", "scale"] as GizmoMode[]).map((m) => (
              <button
                key={m}
                onClick={() => onGizmoModeChange(m)}
                style={styles.gizmoButton(gizmoMode === m)}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      )}

      <div style={styles.buttonRow}>
        <button
          onClick={() => setIsEditing(!isEditing)}
          style={styles.editButton(isEditing)}
        >
          {isEditing ? "Cancel" : "Edit"}
        </button>
        <button onClick={onSave} style={styles.saveButton}>
          Save
        </button>
      </div>

      {saveWarning && <div style={styles.warningBox}>{saveWarning}</div>}

      {onDelete && (
        <div style={styles.deleteSection}>
          <ConfirmDelete onDelete={onDelete} />
        </div>
      )}

      <button onClick={onDismiss} style={styles.dismissButton}>
        Dismiss (Esc)
      </button>
    </div>
  );
}
