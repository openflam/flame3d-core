# Flame3D

## Dependencies

```
conda create -n flame3d-core python=3.12
pip install -r requirements.txt
```

## Process Data

All algorithms in core assume a right-handed Z-up coordinate system. Vendor specific code does whatever is needed to convert vendor data to this coordinate frame.

Algorithms implemented:

- RGB + Depth + Pose + Mesh (e.g., Polycam): Use mesh reprojection to project mesh vertices onto images to find 2D-3D correspondence.
