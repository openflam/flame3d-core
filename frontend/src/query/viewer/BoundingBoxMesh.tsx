import { Billboard, Edges, Text } from "@react-three/drei";
import * as THREE from "three";
import { useMemo } from "react";
import type { BoundingBox } from "../types/global";

interface BoundingBoxMeshProps {
  bbox: BoundingBox;
  color: string | THREE.Color;
  label?: string;
  opacity?: number;
  onClick?: (e: any) => void;
  isSelected?: boolean;
  isDimmed?: boolean;
}

export default function BoundingBoxMesh({
  bbox,
  color,
  label,
  opacity = 0.1,
  onClick,
  isSelected = false,
  isDimmed = false,
}: BoundingBoxMeshProps) {
  const { corners } = bbox;

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const vertices = new Float32Array(24);
    for (let i = 0; i < 8; i++) {
      vertices[i * 3] = corners[i][0];
      vertices[i * 3 + 1] = corners[i][1];
      vertices[i * 3 + 2] = corners[i][2];
    }
    geo.setAttribute("position", new THREE.BufferAttribute(vertices, 3));

    // Standard box indices for 8 corners
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
    return geo;
  }, [corners]);

  // prettier-ignore
  const center = useMemo(() => {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < 8; i++) {
        cx += corners[i][0];
        cy += corners[i][1];
        cz += corners[i][2];
    }
    return [cx / 8, cy / 8, cz / 8];
  }, [corners]);

  // Find max Y for label
  const maxY = useMemo(() => Math.max(...corners.map((c) => c[1])), [corners]);

  const finalOpacity = isDimmed ? 0.05 : isSelected ? 0.6 : opacity;
  const finalColor = color;

  return (
    <group>
      <mesh
        geometry={geometry}
        onClick={(e) => {
          if (e.button === 0) {
            e.stopPropagation();
            if (onClick) onClick(e);
          }
        }}
      >
        <meshStandardMaterial
          color={finalColor}
          transparent
          opacity={finalOpacity}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
        <Edges color={new THREE.Color(finalColor).multiplyScalar(0.5)} />
      </mesh>
      {label && !isDimmed && (
        <Billboard position={[center[0], maxY + 0.1, center[2]]}>
          <Text
            fontSize={0.1}
            color="white"
            anchorX="center"
            anchorY="bottom"
            outlineWidth={0.02}
            outlineColor="black"
          >
            {label}
          </Text>
        </Billboard>
      )}
    </group>
  );
}
