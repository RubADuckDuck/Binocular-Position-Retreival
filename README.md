# OmniPerspect: Real-time 3D Position Tracking and Off-axis Projection

[![Video Demonstration](/images/video_sumnail.png)](https://www.youtube.com/shorts/YI32PZjzeMI)  
[Check out the Blog!](https://rubaduckduck.github.io/2024/07/20/omni_perspect.html)   
[Unity-side Github](https://github.com/RubADuckDuck/OffaxisProjUnity?tab=readme-ov-file)

A binocular camera-based 3D position tracking system that uses two cameras to accurately track objects in three-dimensional space.

## Overview

BNC Tracker (Binocular Camera Tracker) is a Python package that enables real-time 3D position tracking using two standard webcams. By detecting colored markers in both camera feeds, the system calculates 3D coordinates through ray intersection. This provides an affordable alternative to commercial motion capture systems for various applications including:

- Human-computer interaction experiments
- Object tracking for robotics
- Motion analysis for sports or rehabilitation
- Interactive art installations
- DIY motion capture

## Features

- **Real-time tracking**: Process camera feeds at high frame rates
- **Color-based detection**: Track objects based on their color in HSV space
- **3D coordinate calculation**: Determine precise 3D positions using geometric ray intersection
- **Visualization tools**: Real-time 3D plotting of tracking results
- **Multiple communication methods**: Support for WebSockets and shared memory for integration with other applications
- **Camera calibration**: Tools for proper camera setup and alignment

## Installation

### Prerequisites

- Python 3.6 or higher
- Two USB webcams
- OpenCV-compatible cameras

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/bnc_tracker.git
cd bnc_tracker

# Install the package
pip install -e .

# For development tools (testing, formatting)
pip install -e .[dev]
```

## Quick Start

```python
from bnc_tracker import create_tracking_system
from bnc_tracker.core.utils import find_color_coordinates_hsv
from bnc_tracker.tracking.intersection import calculate_intersection_of_ray
import cv2
import numpy as np

# Set up cameras
camera1, camera2 = create_tracking_system(camera1_id=0, camera2_id=1)

try:
    while True:
        # Capture frames
        frame1 = camera1.capture()
        frame2 = camera2.capture()
        
        # Find colored marker in each frame
        position_cam1 = find_color_coordinates_hsv(frame1)
        position_cam2 = find_color_coordinates_hsv(frame2)
        
        if position_cam1 and position_cam2:
            # Calculate 3D position
            # (Add camera position and conversion code here)
            
            # Display the result
            print(f"3D Position: {intersection_point}")
            
        # Display camera feeds
        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    camera1.release()
    camera2.release()
    cv2.destroyAllWindows()
```

For more detailed examples, see the `examples/` directory.

## Usage

### Camera Setup

For optimal tracking:
1. Position cameras at approximately 90° angles from each other
2. Ensure both cameras can see the tracking volume
3. Use consistent lighting to minimize color detection issues
4. Adjust camera parameters (focal length, field of view) in your configuration

### Color Tracking

The system uses HSV color space for more robust color detection. You can adjust the color detection parameters in `bnc_tracker/core/utils.py`:

```python
# Lower and upper HSV bounds for the color you want to track
lower_color = np.array([84, 113, 158])  # Adjust for your marker
upper_color = np.array([179, 255, 255])
```
Before you do so, use 'test/get_hsv_color.py' to get the hsv range for the color of the object you want to detect.   

### Running the Main Application

```bash
# Run the tracker with default settings
bnc-tracker

# Or run the script directly
python -m bnc_tracker.scripts.bnc_tracker_app
```

## Development

### Project Structure

```
bnc_tracker/
├── bnc_tracker/         # Main package
│   ├── core/            # Core functionality
│   ├── tracking/        # Tracking algorithms
│   └── visualization/   # Visualization tools
├── examples/            # Example scripts
├── tests/               # Unit tests
└── archive/            # Previous versions
```

### Running Tests

```bash
# Run the test suite
pytest

# Run with coverage report
pytest --cov=bnc_tracker
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project was inspired by the need for affordable motion capture solutions
- Special thanks to the OpenCV and NumPy communities for their excellent libraries

---
