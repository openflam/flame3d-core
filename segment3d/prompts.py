# fmt: off

# Prompt for captioning images of indidivudal components
CAPTION_IMAGE_PROMPT = \
"""These images show different views of the same object or region in a 3D scene.
Analyze all the images together and provide a concise, descriptive caption
that captures what this object or region is. Focus on:
1. What the main object/region is
2. Its key visual characteristics (color, shape, texture)
3. Any notable features or context

Keep the caption clear and factual, suitable for 3D semantic search."""

# Prompt for captioning a specific object in an image
CAPTION_OBJECT_IN_IMAGE_PROMPT = \
"""These images show different views of a {object_name} in a 3D scene.
Analyze all the images together and provide a concise, descriptive caption
that captures what this {object_name} is. 

Only the properties of the {object_name} should be described. All other objects lying on it, 
around it, or in the background must be ignored and should not be mentioned in the caption.

Focus on:
1. What the {object_name} is
2. Its key visual characteristics (color, shape, texture)
3. Any notable features of the {object_name}

Keep the caption clear and factual, suitable for 3D semantic search."""

IDENTIFY_COMPONENT_PROMPT = \
"""Identify all of the tangible objects in this image. List them in a comma-separated format.
Use common object names. DO NOT include apperance or color details, just the object names. 
For example, say "chair" instead of "red chair" or "wooden chair"."""
