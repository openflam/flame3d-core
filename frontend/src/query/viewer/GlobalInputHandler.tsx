import { useFrame } from "@react-three/fiber";
import { useKeyboardControls } from "@react-three/drei";

interface GlobalInputHandlerProps {
  onExit: () => void;
}

export default function GlobalInputHandler({
  onExit,
}: GlobalInputHandlerProps) {
  const [, get] = useKeyboardControls();
  useFrame(() => {
    if (get().escape) onExit();
  });
  return null;
}
