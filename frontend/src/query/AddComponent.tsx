import { useState } from "react";
import { styles } from "./addComponentStyles";

import type { GizmoMode } from "./ComponentDetails";
import type { BoundingBox } from "./types/global";
import { addComponent } from "./query";

interface AddComponentProps {
  onDismiss: () => void;
  gizmoMode: GizmoMode;
  onGizmoModeChange: (mode: GizmoMode) => void;
  datasetName: string;
  editedBBox: BoundingBox | null;
}

export default function AddComponent({ onDismiss, gizmoMode, onGizmoModeChange, datasetName, editedBBox }: AddComponentProps) {
  const [description, setDescription] = useState("");
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div style={styles.panel}>
      <h3 style={styles.sectionLabel}>Add New Component</h3>

      <div style={styles.fieldContainer}>
        <label style={styles.fieldLabel}>Description</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          style={styles.descriptionTextarea}
          placeholder="Enter description..."
        />
      </div>

      <div style={styles.fieldContainer}>
        <label style={styles.fieldLabel}>Upload Image</label>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          style={styles.fileInput}
        />
      </div>

      {imagePreview && (
        <div style={styles.imageWrapper}>
          <img src={imagePreview} alt="Preview" style={styles.image} />
        </div>
      )}

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

      <div style={styles.buttonRow}>
        <button onClick={onDismiss} style={styles.editButton(false)}>
          Cancel
        </button>
        <button
          style={styles.saveButton}
          onClick={async () => {
            if (!editedBBox) return;
            try {
              const bboxToSave = {
                corners: editedBBox.corners
              };
              let base64Data: string | null = null;
              if (imagePreview) {
                // Strip the "data:image/jpeg;base64," prefix
                base64Data = imagePreview.split(',')[1] || null;
              }
              await addComponent(datasetName, description, bboxToSave, base64Data);
              onDismiss();
            } catch (err) {
              console.error(err);
              alert("Failed to add component.");
            }
          }}
        >
          Add
        </button>
      </div>
    </div>
  );
}
