import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, KeyboardControls } from "@react-three/drei";
import { Suspense, useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import type { BoundingBox, Route } from "./types/global";
import ComponentDetails from "./ComponentDetails";
import AddComponent from "./AddComponent";
import { styles as addComponentStyles } from "./addComponentStyles";
import type { GizmoMode } from "./ComponentDetails";
import { viewerToWorldBbox } from "./transforms";
import Model from "./viewer/Model";
import BoundingBoxMesh from "./viewer/BoundingBoxMesh";
import CameraController from "./viewer/CameraController";
import GlobalInputHandler from "./viewer/GlobalInputHandler";
import BBoxTransformGizmo from "./viewer/BBoxTransformGizmo";
import RoutePath from "./viewer/RoutePath";

interface Model3DViewerProps {
  source: string;
  boundingBox?: BoundingBox[];
  captions?: string[];
  componentIds?: string[];
  componentColors?: string[];
  autoTagBBoxes?: BoundingBox[];
  showAutoTags?: boolean;
  occupancyGrid?: BoundingBox[];
  showOccupancyGrid?: boolean;
  annotations?: string[];
  route?: Route;
  datasetName: string;
  focusedBBoxIndex?: number | null;
  externalSelectedBBoxIndex?: number | null;
}

/** Moves the camera to frame the given bounding box when it changes. */
function FocusBBoxController({ bbox }: { bbox: BoundingBox | null }) {
  const { camera, controls } = useThree();

  useEffect(() => {
    if (!bbox || !controls) return;
    const center = new THREE.Vector3();
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;

    for (const p of bbox.corners) {
      center.x += p[0];
      center.y += p[1];
      center.z += p[2];
      minX = Math.min(minX, p[0]);
      minY = Math.min(minY, p[1]);
      minZ = Math.min(minZ, p[2]);
      maxX = Math.max(maxX, p[0]);
      maxY = Math.max(maxY, p[1]);
      maxZ = Math.max(maxZ, p[2]);
    }
    center.divideScalar(8);

    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const distance = Math.max(size * 2.5, 0.5);

    // Keep the current camera direction but re-anchor it on the new center
    const orbitControls = controls as any;
    const camDir = new THREE.Vector3()
      .subVectors(camera.position, orbitControls.target)
      .normalize();
    if (camDir.lengthSq() < 0.001) camDir.set(1, 0.5, 1).normalize();

    camera.position.copy(center).addScaledVector(camDir, distance);
    orbitControls.target.copy(center);
    orbitControls.update();
  }, [bbox]); // eslint-disable-line react-hooks/exhaustive-deps

  return null;
}

const keyboardMap = [
  { name: "forward", keys: ["ArrowUp", "KeyW"] },
  { name: "backward", keys: ["ArrowDown", "KeyS"] },
  { name: "left", keys: ["ArrowLeft", "KeyA"] },
  { name: "right", keys: ["ArrowRight", "KeyD"] },
  { name: "escape", keys: ["Escape"] },
];

export default function Model3DViewer({
  source,
  boundingBox,
  captions,
  componentIds,
  componentColors,
  autoTagBBoxes,
  showAutoTags,
  occupancyGrid,
  showOccupancyGrid,
  annotations,
  route,
  datasetName,
  focusedBBoxIndex,
  externalSelectedBBoxIndex,
}: Model3DViewerProps) {
  const [selectedBBoxIndex, setSelectedBBoxIndex] = useState<number | null>(
    null,
  );
  const [selectedAutoTagId, setSelectedAutoTagId] = useState<string | null>(
    null,
  );

  // Sync external selection (e.g. from the component list sidebar) into the
  // internal selectedBBoxIndex so ComponentDetails is shown automatically.
  useEffect(() => {
    if (externalSelectedBBoxIndex !== undefined) {
      setSelectedBBoxIndex(externalSelectedBBoxIndex ?? null);
      setSelectedAutoTagId(null);
    }
  }, [externalSelectedBBoxIndex]);
  const [isEditing, setIsEditing] = useState(false);
  const [editedCaption, setEditedCaption] = useState("");
  const [componentImage, setComponentImage] = useState<string | null>(null);
  const [isLoadingComponent, setIsLoadingComponent] = useState(false);
  const [editedBBox, setEditedBBox] = useState<BoundingBox | null>(null);
  const [gizmoMode, setGizmoMode] = useState<GizmoMode>("translate");
  // savedBBoxMap stores saved positions keyed by component ID so the box
  // stays in its new position even after the panel is dismissed/deselected.
  const [savedBBoxMap, setSavedBBoxMap] = useState<Map<string, BoundingBox>>(
    new Map(),
  );
  const [saveWarning, setSaveWarning] = useState<string | null>(null);
  const [deletedIds, setDeletedIds] = useState<Set<string>>(new Set());
  const [isAddingComponent, setIsAddingComponent] = useState(false);

  const handleAddComponentClick = () => {
    setIsAddingComponent(true);
    handleDeselect();
    const s = 0.5;
    setEditedBBox({
      corners: [
        [-s, -s, -s],
        [s, -s, -s],
        [s, s, -s],
        [-s, s, -s],
        [-s, -s, s],
        [s, -s, s],
        [s, s, s],
        [-s, s, s],
      ]
    });
    setIsEditing(true);
  };

  // Disable OrbitControls as soon as the gizmo is visible so it never
  // interferes with pointer events intended for TransformControls.
  const orbitEnabled = !(isEditing && editedBBox !== null);

  const handleDeselect = () => {
    setSelectedBBoxIndex(null);
    setSelectedAutoTagId(null);
  };

  // Derive the bbox of whichever item is currently selected, preferring any
  // previously saved position so the gizmo starts from the right place if
  // the user opens edit mode again after a save.
  const selectedBBoxData: BoundingBox | null = useMemo(() => {
    let rawBBox: BoundingBox | null = null;
    let key: string | null = null;
    if (selectedBBoxIndex !== null && boundingBox) {
      rawBBox = boundingBox[selectedBBoxIndex] ?? null;
      key = componentIds?.[selectedBBoxIndex] ?? null;
    } else if (selectedAutoTagId !== null && annotations && autoTagBBoxes) {
      const idx = annotations.indexOf(selectedAutoTagId);
      rawBBox = idx >= 0 ? (autoTagBBoxes[idx] ?? null) : null;
      key = selectedAutoTagId;
    }
    if (key && savedBBoxMap.has(key)) return savedBBoxMap.get(key)!;
    return rawBBox;
  }, [
    selectedBBoxIndex,
    selectedAutoTagId,
    boundingBox,
    annotations,
    autoTagBBoxes,
    componentIds,
    savedBBoxMap,
  ]);

  // Initialise editedBBox when entering edit mode
  useEffect(() => {
    if (isAddingComponent) return;
    if (isEditing) {
      setEditedBBox(selectedBBoxData);
    } else {
      setEditedBBox(null);
    }
  }, [isEditing, selectedBBoxData, isAddingComponent]);

  // Clear the save warning whenever the selection changes
  useEffect(() => {
    setSaveWarning(null);
  }, [selectedBBoxIndex, selectedAutoTagId]);

  const handleSave = async () => {
    // Capture before setIsEditing(false) clears editedBBox via useEffect
    const bboxToSave = editedBBox;
    const captionToSave = editedCaption;
    // Persist the new position immediately so the box doesn't snap back when
    // isEditing → false unmounts the gizmo and restores the BoundingBoxMesh.
    if (bboxToSave && currentComponentId) {
      setSavedBBoxMap((prev) =>
        new Map(prev).set(currentComponentId, bboxToSave),
      );
    }
    setIsEditing(false);
    if (!currentComponentId) return;

    if (currentComponentId.startsWith("custom_bbox_")) {
      setSaveWarning("Cannot save custom locations to the database.");
      return;
    }

    try {
      const { updateComponent } = await import("./query");
      const updates: {
        caption: string;
        bbox?: { corners: [number, number, number][] };
      } = {
        caption: captionToSave,
      };
      if (bboxToSave) {
        // The gizmo edits in the viewer frame; convert to the world frame for the server.
        updates.bbox = viewerToWorldBbox(bboxToSave);
      }
      await updateComponent(currentComponentId, updates, datasetName);
      setSaveWarning(null);
    } catch (error) {
      console.error("Error saving component:", error);
      setSaveWarning("Save failed — changes may not have been persisted.");
    }
  };

  const currentComponentId: string | null = useMemo(() => {
    if (selectedAutoTagId !== null) return selectedAutoTagId;
    if (selectedBBoxIndex !== null && componentIds)
      return componentIds[selectedBBoxIndex] ?? null;
    return null;
  }, [selectedAutoTagId, selectedBBoxIndex, componentIds]);

  useEffect(() => {
    const fetchComponentInfo = async () => {
      // Handle autotag selection
      if (selectedAutoTagId !== null) {
        setIsLoadingComponent(true);
        try {
          const { getComponentInfo } = await import("./query");
          const info = await getComponentInfo(selectedAutoTagId, datasetName);
          setEditedCaption(info.caption || "");
          setComponentImage(info.image_base64);
        } catch (error) {
          console.error("Error fetching component info:", error);
          setEditedCaption("Error loading component information");
          setComponentImage(null);
        } finally {
          setIsLoadingComponent(false);
        }
      }
      // Handle search result selection
      else if (selectedBBoxIndex !== null && componentIds) {
        const componentId = componentIds[selectedBBoxIndex];
        if (componentId && !componentId.startsWith("custom_bbox_")) {
          setIsLoadingComponent(true);
          try {
            const { getComponentInfo } = await import("./query");
            const info = await getComponentInfo(componentId, datasetName);
            setEditedCaption(info.caption || "");
            setComponentImage(info.image_base64);
          } catch (error) {
            console.error("Error fetching component info:", error);
            // Fallback to captions array if API fails
            if (captions) {
              setEditedCaption(captions[selectedBBoxIndex] || "");
            }
            setComponentImage(null);
          } finally {
            setIsLoadingComponent(false);
          }
        } else if (captions) {
          setEditedCaption(captions[selectedBBoxIndex] || "");
          setComponentImage(null);
        }
      }
      setIsEditing(false);
    };

    fetchComponentInfo();
  }, [selectedBBoxIndex, selectedAutoTagId, captions, componentIds]);

  const hasSelection = selectedBBoxIndex !== null || selectedAutoTagId !== null;

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        overflow: "hidden",
        fontFamily: "sans-serif",
      }}
    >
      <KeyboardControls map={keyboardMap}>
        {/* --- HTML OVERLAY --- */}
        {!isAddingComponent && (
          <button
            onClick={handleAddComponentClick}
            style={addComponentStyles.openButton}
          >
            Add Component +
          </button>
        )}

        {isAddingComponent && (
          <AddComponent
            onDismiss={() => {
              setIsAddingComponent(false);
              setIsEditing(false);
              setEditedBBox(null);
            }}
            gizmoMode={gizmoMode}
            onGizmoModeChange={setGizmoMode}
            datasetName={datasetName}
            editedBBox={editedBBox}
          />
        )}

        {hasSelection && !isAddingComponent && (
          <ComponentDetails
            editedCaption={editedCaption}
            setEditedCaption={setEditedCaption}
            isEditing={isEditing}
            setIsEditing={setIsEditing}
            onDismiss={handleDeselect}
            onSave={handleSave}
            onDelete={
              currentComponentId && !currentComponentId.startsWith("custom_bbox_")
                ? async () => {
                    try {
                      const { deleteComponent } = await import("./query");
                      await deleteComponent(currentComponentId, datasetName);
                      setDeletedIds((prev) =>
                        new Set(prev).add(currentComponentId),
                      );
                      handleDeselect();
                    } catch (error) {
                      console.error("Error deleting component:", error);
                    }
                  }
                : undefined
            }
            saveWarning={saveWarning}
            componentId={currentComponentId}
            imageBase64={componentImage}
            isLoading={isLoadingComponent}
            editedBBox={editedBBox}
            gizmoMode={gizmoMode}
            onGizmoModeChange={setGizmoMode}
          />
        )}

        {/* --- 3D SCENE --- */}
        <Canvas
          camera={{ position: [5, 5, 5] }}
          /* Removed onPointerMissed to prevent deselection on background click */
        >
          <ambientLight intensity={2.0} />
          <pointLight position={[10, 10, 10]} intensity={2.0} />
          <OrbitControls makeDefault enabled={orbitEnabled} />
          <CameraController />
          <GlobalInputHandler onExit={handleDeselect} />
          <FocusBBoxController
            bbox={
              focusedBBoxIndex != null && boundingBox
                ? (boundingBox[focusedBBoxIndex] ?? null)
                : null
            }
          />

          <Suspense fallback={null}>
            <Model url={source} />
          </Suspense>

          {boundingBox?.map((bbox, i) => {
            const savedId = componentIds?.[i] ?? null;
            if (savedId && deletedIds.has(savedId)) return null;
            const isThisSelected =
              selectedBBoxIndex === i && selectedAutoTagId === null;
            if (isThisSelected && isEditing) return null;
            const displayBBox =
              savedId && savedBBoxMap.has(savedId)
                ? savedBBoxMap.get(savedId)!
                : bbox;
            const color = componentColors?.[i] || "red";
            return (
              <BoundingBoxMesh
                key={`bbox-${i}`}
                bbox={displayBBox}
                color={color}
                opacity={0.3}
                onClick={() => {
                  setSelectedBBoxIndex(i);
                  setSelectedAutoTagId(null);
                }}
                isSelected={isThisSelected}
                isDimmed={
                  (selectedBBoxIndex !== null && selectedBBoxIndex !== i) ||
                  selectedAutoTagId !== null
                }
              />
            );
          })}

          {showAutoTags &&
            autoTagBBoxes?.map((bbox, i) => {
              const annotId = annotations?.[i] ?? null;
              if (annotId && deletedIds.has(annotId)) return null;
              const isThisSelected = selectedAutoTagId === annotations?.[i];
              if (isThisSelected && isEditing) return null;
              const displayBBox =
                annotId && savedBBoxMap.has(annotId)
                  ? savedBBoxMap.get(annotId)!
                  : bbox;
              return (
                <BoundingBoxMesh
                  key={`autotag-${i}`}
                  bbox={displayBBox}
                  color={new THREE.Color().setHSL((i * 0.618) % 1, 0.8, 0.5)}
                  label={annotations?.[i]}
                  opacity={0.1}
                  onClick={() => {
                    if (annotations?.[i]) {
                      setSelectedAutoTagId(annotations[i]);
                      setSelectedBBoxIndex(null);
                    }
                  }}
                  isSelected={isThisSelected}
                  isDimmed={
                    selectedBBoxIndex !== null ||
                    (selectedAutoTagId !== null &&
                      selectedAutoTagId !== annotations?.[i])
                  }
                />
              );
            })}

          {showOccupancyGrid &&
            occupancyGrid?.map((bbox, i) => (
              <BoundingBoxMesh
                key={`occupancy-${i}`}
                bbox={bbox}
                color="green"
                opacity={0.1}
                isDimmed={selectedBBoxIndex !== null}
              />
            ))}

          {route && <RoutePath route={route} />}

          {isEditing && editedBBox && (
            <BBoxTransformGizmo
              key="bbox-gizmo"
              initialBBox={editedBBox}
              mode={gizmoMode}
              onCommit={setEditedBBox}
            />
          )}
        </Canvas>
      </KeyboardControls>
    </div>
  );
}
