import { useEffect, useRef, useState } from "react";
import { Spinner } from "react-bootstrap";
import { getComponentInfo } from "./query";
import { styles } from "./searchComponentListStyles";

interface SearchComponentListProps {
  componentIds: string[];
  captions: string[];
  datasetName: string;
  onComponentClick: (index: number) => void;
  focusedComponentIndex: number | null;
}

interface ComponentEntry {
  index: number;
  imageBase64: string | null;
  caption: string;
  loaded: boolean;
}

export default function SearchComponentList({
  componentIds,
  captions,
  datasetName,
  onComponentClick,
  focusedComponentIndex,
}: SearchComponentListProps) {
  const [entries, setEntries] = useState<ComponentEntry[]>([]);
  const fetchKeyRef = useRef(0);

  useEffect(() => {
    if (componentIds.length === 0) {
      setEntries([]);
      return;
    }

    // Seed with loading placeholders immediately
    setEntries(
      componentIds.map((_, i) => ({
        index: i,
        imageBase64: null,
        caption: captions[i] ?? "",
        loaded: false,
      })),
    );

    // Track a fetch generation so stale responses from previous searches are
    // discarded when a new search fires before all images arrive.
    const generation = ++fetchKeyRef.current;

    componentIds.forEach((id, i) => {
      if (id.startsWith("custom_bbox_")) {
        setEntries((prev) =>
          prev.map((e) => (e.index === i ? { ...e, loaded: true } : e)),
        );
        return;
      }
      getComponentInfo(id, datasetName)
        .then((info) => {
          if (fetchKeyRef.current !== generation) return;
          setEntries((prev) =>
            prev.map((e) =>
              e.index === i
                ? { ...e, imageBase64: info.image_base64, loaded: true }
                : e,
            ),
          );
        })
        .catch(() => {
          if (fetchKeyRef.current !== generation) return;
          setEntries((prev) =>
            prev.map((e) => (e.index === i ? { ...e, loaded: true } : e)),
          );
        });
    });
  }, [componentIds, datasetName]); // eslint-disable-line react-hooks/exhaustive-deps

  if (componentIds.length === 0) return null;

  return (
    <div style={styles.root}>
      {/* Scrollable image list */}
      <div style={styles.list}>
        {entries.map((entry) => {
          const isFocused = focusedComponentIndex === entry.index;
          const compId = componentIds[entry.index];
          return (
            <div
              key={entry.index}
              onClick={() => onComponentClick(entry.index)}
              title={entry.caption}
              style={{
                ...styles.card,
                ...(isFocused
                  ? styles.cardBorderFocused
                  : styles.cardBorderDefault),
              }}
            >
              {!entry.loaded ? (
                <Spinner
                  animation="border"
                  size="sm"
                  variant="secondary"
                  role="status"
                />
              ) : entry.imageBase64 ? (
                <img
                  src={`data:image/jpeg;base64,${entry.imageBase64}`}
                  alt={entry.caption}
                  style={styles.image}
                />
              ) : (
                <span style={styles.noImage}>No image</span>
              )}
              {/* Component ID badge — top-left */}
              {compId && <div style={styles.componentIdBadge}>{compId}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
