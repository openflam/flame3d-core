"""
vLLM-based object identifier implementation using HuggingFace AutoProcessor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from ..prompts import IDENTIFY_COMPONENT_PROMPT
from .identifier_base import IdentificationResult


class VLLMIdentifier:
    """
    Object identifier implementation using vLLM for inference and HuggingFace
    AutoProcessor for input formatting.

    For each frame, it sends the image to the VLM with IDENTIFY_COMPONENT_PROMPT
    and parses the comma-separated response into a list of objects.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: int = 0,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "bfloat16",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        """
        Initialize the vLLM identifier.

        Args:
            model: HuggingFace model name
            device: GPU device ID
            max_model_len: Maximum model context length
            gpu_memory_utilization: GPU memory utilization for vLLM
            dtype: Data type for model weights
            max_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature (0.0 for deterministic)
        """
        from transformers import AutoProcessor
        from vllm import LLM

        self.model = model
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature

        print(f"\nInitializing AutoProcessor for {model}...")
        self.processor = AutoProcessor.from_pretrained(model)
        print("Processor loaded successfully!")

        print(f"\nInitializing vLLM with {model}...")
        self.llm = LLM(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print("Model loaded successfully!")

    def _prepare_frame_input(
        self,
        frame_name: str,
        image_path: Path,
    ) -> tuple[str, dict | None]:
        """
        Prepare vLLM input for a single frame.

        Args:
            frame_name: Name/identifier of the frame
            image_path: Path to the image file

        Returns:
            Tuple of (image_path_str, vllm_input) or (path_str, None) if image missing.
            vllm_input is a dict with 'prompt' and 'multi_modal_data' keys.
        """
        from PIL import Image

        image_path_str = str(image_path)

        if not image_path.exists():
            print(f"  Warning: Image not found: {image_path}, skipping")
            return image_path_str, None

        pil_image = Image.open(image_path)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": IDENTIFY_COMPONENT_PROMPT},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vllm_input = {
            "prompt": prompt,
            "multi_modal_data": {"image": [pil_image]},
        }

        return image_path_str, vllm_input

    def _parse_objects(self, raw_text: str) -> List[str]:
        """
        Parse a comma-separated list of objects from model output.

        Args:
            raw_text: Raw text response from the model

        Returns:
            List of object name strings (stripped and non-empty)
        """
        objects = [obj.strip() for obj in raw_text.split(",")]
        objects = list(set(objects))  # Remove duplicates
        objects = [obj for obj in objects if obj]  # Remove empty strings
        return [obj for obj in objects if obj]

    def identify_batch(
        self,
        batch_data: List[tuple[str, Path]],
    ) -> List[IdentificationResult]:
        """
        Identify objects in a batch of frames using vLLM.

        Args:
            batch_data: List of (frame_name, image_path) tuples

        Returns:
            List of IdentificationResult objects
        """
        from vllm import SamplingParams

        results = []
        vllm_inputs = []
        valid_frame_names = []
        valid_image_paths = []

        # Prepare all inputs
        for frame_name, image_path in batch_data:
            image_path_str, vllm_input = self._prepare_frame_input(
                frame_name, image_path
            )

            if vllm_input is not None:
                vllm_inputs.append(vllm_input)
                valid_frame_names.append(frame_name)
                valid_image_paths.append(image_path_str)

        # Process batch with vLLM
        if vllm_inputs:
            try:
                sampling_params = SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                outputs = self.llm.generate(
                    vllm_inputs,
                    sampling_params=sampling_params,
                )

                for frame_name, output, image_path_str in zip(
                    valid_frame_names, outputs, valid_image_paths
                ):
                    raw_text = output.outputs[0].text.strip()
                    objects = self._parse_objects(raw_text)

                    results.append(
                        IdentificationResult(
                            frame_name=frame_name,
                            objects=objects,
                            image_path=image_path_str,
                            error=None,
                        )
                    )

            except Exception as e:
                print(f"  Error calling vLLM for batch: {e}")
                for frame_name, image_path_str in zip(
                    valid_frame_names, valid_image_paths
                ):
                    results.append(
                        IdentificationResult(
                            frame_name=frame_name,
                            objects=[],
                            image_path=image_path_str,
                            error=str(e),
                        )
                    )

        return results

    def cleanup(self) -> None:
        """Clean up resources (vLLM handles this automatically)."""
        pass
