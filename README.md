# Flame3D

## Getting Started

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed

### Build & Run

```bash
# Build the Docker image (only needed once, or when dependencies change)
docker compose build

# Start the server
docker compose up

# Start in detached mode
docker compose up -d
```

### Test

```bash
# Health check
curl http://localhost:5005/health

# Verify conda environments
docker compose exec flame3d-core conda env list
```

### Development

Code changes are volume-mounted into the container — just restart to pick them up:

```bash
docker compose restart
```

## Process Data

All algorithms in core assume a right-handed Z-up coordinate system. Vendor specific code does whatever is needed to convert vendor data to this coordinate frame.

Algorithms implemented:

- RGB + Depth + Pose + Mesh (e.g., Polycam): Use mesh reprojection to project mesh vertices onto images to find 2D-3D correspondence.
