# Image Stream
• Gets images as python iterators from folders, video files, and webcams
• Adds marks and text to images
• Shows images
• Overlays masks onto images to visualize classification

# Kalman Filter
• A general-purpose Kalman Filter class
• Auto-fills values that are not specified
• Combines readings from different sensors
• If no dt is specified, it times iterations of the loop, and recalculates parameters with that dt

# Visual Odometry
• Estimates the camera transform between images (monocular)
• Provides the transform between pairs of images, as well as the total transform so far
• Converts the rotation matrix into roll, pitch, yaw euler angles
• Implements the kalman filter in the example
