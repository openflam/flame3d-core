"""
vLLM-based captioner implementation using HuggingFace AutoProcessor.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from ..prompts import CAPTION_IMAGE_PROMPT, CAPTION_OBJECT_IN_IMAGE_PROMPT
from .captioner_base import CaptionResult


class VLLMCaptioner:
    """
    Captioner implementation using vLLM for inference and HuggingFace AutoProcessor
    for input formatting.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: int = 0,
        max_model_len: int = 4096,
        max_images_per_prompt: int = 10,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "bfloat16",
        max_tokens: int = 300,
        temperature: float = 0.0,
    ):
        """
        Initialize the vLLM captioner.

        Args:
            model: HuggingFace model name
            device: GPU device ID
            max_model_len: Maximum model context length
            max_images_per_prompt: Maximum number of images per prompt
            gpu_memory_utilization: GPU memory utilization for vLLM
            dtype: Data type for model weights
            max_tokens: Maximum tokens to generate per caption
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
            limit_mm_per_prompt={"image": max_images_per_prompt},
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print("Model loaded successfully!")

    def _prepare_component_messages(
        self,
        component_id: str | int,
        top_images: List[Dict[str, Any]],
        crops_dir: Path,
    ) -> tuple[List[str], Optional[Dict]]:
        """
        Prepare messages for a component in vLLM format using AutoProcessor.

        Args:
            component_id: ID of the component
            top_images: List of top crop information
            crops_dir: Directory containing the cropped images

        Returns:
            Tuple of (image_paths, vllm_input) or ([], None) if no valid images
            vllm_input is a dict with 'prompt' and 'multi_modal_data' keys
        """
        from PIL import Image

        # Prepare image paths from crop filenames
        image_paths = []
        pil_images = []

        for crop_info in top_images:
            crop_filename = crop_info["crop_filename"]
            image_path = crops_dir / f"component_{component_id}" / crop_filename

            if not image_path.exists():
                print(f"  Warning: Crop image not found: {image_path}, skipping")
                continue

            image_paths.append(str(image_path))
            # Load image as PIL Image
            pil_images.append(Image.open(image_path))

        if not image_paths:
            return [], None

        # Extract object name if present
        object_name = None
        for crop_info in top_images:
            crop_filename = crop_info["crop_filename"]
            # Extract object name from patterns like "<prefix>_<object>_seq_..."
            # e.g. "frame_00161_stool_seq_1_5_masked_crop.jpg" → "stool"
            #      "50927146410_box_seq_6_3_masked_crop.jpg"    → "box"
            match = re.search(r'.+_(.+?)_seq_', crop_filename)
            if match:
                object_name = match.group(1).replace('_', ' ')
                break

        # Create messages in the format expected by AutoProcessor
        if object_name:
            question = CAPTION_OBJECT_IN_IMAGE_PROMPT.format(object_name=object_name)
        else:
            question = CAPTION_IMAGE_PROMPT

        # Build placeholders for each image
        placeholders = [{"type": "image", "image": img} for img in pil_images]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": question},
                ],
            },
        ]

        # Use AutoProcessor to format the prompt
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Build vLLM input format with proper structure
        vllm_input = {
            "prompt": prompt,
            "multi_modal_data": {"image": pil_images},
        }

        return image_paths, vllm_input

    def _cleanup_caption(self, caption: str) -> str:
        """
        Clean up model-specific quirks in generated captions.

        Args:
            caption: Raw caption from model

        Returns:
            Cleaned caption
        """
        # Qwen/Qwen2.5-VL-7B-Instruct generates "addCriterion" for some reason.
        # Just remove that if it appears.
        if self.model == "Qwen/Qwen2.5-VL-7B-Instruct":
            caption = caption.replace("addCriterion:", "").strip()
            caption = caption.replace("addCriterion\n", "").strip()
            caption = caption.replace("addCriterion", "").strip()
            caption = caption.strip('"\\"')

        return caption

    def caption_batch(
        self,
        batch_data: List[tuple[str | int, List[Dict[str, Any]]]],
        crops_dir: Path,
    ) -> List[CaptionResult]:
        """
        Generate captions for multiple components in batch using vLLM.

        Args:
            batch_data: List of (component_id, top_images) tuples
            crops_dir: Directory containing the cropped images

        Returns:
            List of CaptionResult objects
        """
        from vllm import SamplingParams

        results = []
        vllm_inputs = []
        batch_component_ids = []
        batch_image_paths = []

        # Prepare all messages
        for component_id, top_images in batch_data:
            image_paths, vllm_input = self._prepare_component_messages(
                component_id, top_images, crops_dir
            )

            if vllm_input is not None:
                vllm_inputs.append(vllm_input)
                batch_component_ids.append(component_id)
                batch_image_paths.append(image_paths)

        # Process batch with vLLM
        if vllm_inputs:
            try:
                # Create SamplingParams object
                sampling_params = SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                # Generate captions in batch
                outputs = self.llm.generate(
                    vllm_inputs,
                    sampling_params=sampling_params,
                )

                # Extract captions from outputs
                for component_id, output, image_paths in zip(
                    batch_component_ids, outputs, batch_image_paths
                ):
                    caption = output.outputs[0].text.strip()
                    caption = self._cleanup_caption(caption)

                    results.append(
                        CaptionResult(
                            component_id=component_id,
                            caption=caption,
                            image_paths=image_paths,
                            error=None,
                        )
                    )

            except Exception as e:
                print(f"  Error calling vLLM for batch: {e}")
                # Add error results for all components in batch
                for component_id, image_paths in zip(
                    batch_component_ids, batch_image_paths
                ):
                    results.append(
                        CaptionResult(
                            component_id=component_id,
                            caption=f"[Error generating caption: {str(e)}]",
                            image_paths=image_paths,
                            error=str(e),
                        )
                    )

        return results

    def cleanup(self) -> None:
        """Clean up resources (vLLM handles this automatically)."""
        pass
