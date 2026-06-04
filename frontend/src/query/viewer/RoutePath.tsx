import { useMemo } from "react";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import type { Route } from "../types/global";

export default function RoutePath({ route }: { route: Route }) {
  if (!route || route.length < 2) return null;
  const points = useMemo(
    () => route.map((pt) => new THREE.Vector3(pt[0], pt[1], pt[2])),
    [route],
  );
  return <Line points={points} color="blue" lineWidth={3} />;
}
