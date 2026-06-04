import { useEffect, useMemo, useRef } from "react";
import { TransformControls, Edges } from "@react-three/drei";
import * as THREE from "three";
import type { BoundingBox } from "../types/global";
import type { GizmoMode } from "../ComponentDetails";

interface BBoxTransformGizmoProps {
  initialBBox: BoundingBox;
  mode: GizmoMode;
  onCommit: (bbox: BoundingBox) => void;
}

export default function BBoxTransformGizmo({
  initialBBox,
  mode,
  onCommit,
}: BBoxTransformGizmoProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const tcRef = useRef<any>(null);
  // Track live transforms in a ref – no React state updates during drag
  const liveBBoxRef = useRef<BoundingBox>(initialBBox);

  // Compute position and geometry declaratively based on the initial box
  const { position, geometry } = useMemo(() => {
    const { corners } = initialBBox;

    // prettier-ignore
    let cx = 0, cy = 0, cz = 0;
    for (const c of corners) {
      cx += c[0];
      cy += c[1];
      cz += c[2];
    }
    cx /= 8;
    cy /= 8;
    cz /= 8;

    const geo = new THREE.BufferGeometry();
    const vertices = new Float32Array(24);
    for (let i = 0; i < 8; i++) {
      vertices[i * 3] = corners[i][0] - cx;
      vertices[i * 3 + 1] = corners[i][1] - cy;
      vertices[i * 3 + 2] = corners[i][2] - cz;
    }
    geo.setAttribute("position", new THREE.BufferAttribute(vertices, 3));

    // prettier-ignore
    const indices = [
      0, 1, 2,  0, 2, 3, // Bottom
      4, 5, 6,  4, 6, 7, // Top
      0, 1, 5,  0, 5, 4, // Front
      3, 2, 6,  3, 6, 7, // Back
      0, 3, 7,  0, 7, 4, // Left
      1, 2, 6,  1, 6, 5  // Right
    ];
    geo.setIndex(indices);
    geo.computeVertexNormals();

    return { position: new THREE.Vector3(cx, cy, cz), geometry: geo };
  }, []); // run once on mount

  // Wire raw Three.js events so drag tracking never triggers React re-renders
  useEffect(() => {
    const tc = tcRef.current;
    if (!tc) return;

    const onObjectChange = () => {
      const m = meshRef.current;
      if (!m) return;

      const geom = m.geometry;
      const posAttr = geom.getAttribute("position");
      const newCorners: [number, number, number][] = [];
      m.updateMatrixWorld(true);

      for (let i = 0; i < 8; i++) {
        const v = new THREE.Vector3(
          posAttr.getX(i),
          posAttr.getY(i),
          posAttr.getZ(i),
        );
        v.applyMatrix4(m.matrixWorld);
        newCorners.push([v.x, v.y, v.z]);
      }

      liveBBoxRef.current = { corners: newCorners };
    };

    const onDraggingChanged = (e: any) => {
      if (!e.value) onCommit(liveBBoxRef.current);
    };

    tc.addEventListener("objectChange", onObjectChange);
    tc.addEventListener("dragging-changed", onDraggingChanged);
    return () => {
      tc.removeEventListener("objectChange", onObjectChange);
      tc.removeEventListener("dragging-changed", onDraggingChanged);
    };
  }, [onCommit]);

  // Render the mesh and TransformControls as siblings.
  // Using the `object` prop (instead of children wrapping) makes TransformControls
  // attach() directly to the mesh, so the gizmo lives at the mesh's world position
  // and moves with it. The children-wrapper approach puts kids inside an internal
  // group at origin, causing the gizmo to appear at (0,0,0).
  return (
    <>
      <mesh ref={meshRef} geometry={geometry} position={position}>
        <meshStandardMaterial
          color="#3b82f6"
          transparent
          opacity={0.35}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
        <Edges color="#1d4ed8" />
      </mesh>
      <TransformControls
        ref={tcRef}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        object={meshRef as any}
        mode={mode}
      />
    </>
  );
}
