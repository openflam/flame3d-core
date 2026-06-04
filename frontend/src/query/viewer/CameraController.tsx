import { useFrame, useThree } from "@react-three/fiber";
import { useKeyboardControls } from "@react-three/drei";
import * as THREE from "three";

export default function CameraController() {
  const [, get] = useKeyboardControls();
  const { camera } = useThree();

  useFrame((state, delta) => {
    const activeElement = document.activeElement;
    if (activeElement && ["INPUT", "TEXTAREA"].includes(activeElement.tagName))
      return;

    const { forward, backward, left, right } = get();
    if (!forward && !backward && !left && !right) return;

    const speed = 5 * delta;
    const direction = new THREE.Vector3();
    if (forward) direction.z -= 1;
    if (backward) direction.z += 1;
    if (left) direction.x -= 1;
    if (right) direction.x += 1;
    if (direction.lengthSq() === 0) return;

    direction
      .normalize()
      .multiplyScalar(speed)
      .applyQuaternion(camera.quaternion);
    camera.position.add(direction);

    const controls = state.controls as any;
    if (controls) controls.target.add(direction);
  });

  return null;
}
