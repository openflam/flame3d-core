import { useGLTF } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

export default function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);

  useMemo(() => {
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        if (child.material) {
          const materials = Array.isArray(child.material)
            ? child.material
            : [child.material];

          materials.forEach((mat) => {
            if (
              mat instanceof THREE.MeshStandardMaterial ||
              mat instanceof THREE.MeshPhysicalMaterial
            ) {
              // Tone down metalness to prevent pure black reflection rendering
              mat.metalness = Math.min(mat.metalness, 0.2);
              mat.roughness = Math.max(mat.roughness, 0.8);

              // Enable vertex colors only if no texture map exists and colors are present
              if (
                !mat.map &&
                child.geometry &&
                child.geometry.attributes.color
              ) {
                mat.vertexColors = true;
                mat.color.setHex(0xffffff); // Ensure base color doesn't darken vertex colors
              }

              mat.needsUpdate = true;
            }
          });
        }
      }
    });
  }, [scene]);

  return <primitive object={scene} />;
}
