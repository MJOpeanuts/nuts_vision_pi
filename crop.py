#!/usr/bin/env python3
"""
Component Cropping Utility
Crops detected components from images and saves them to individual files.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json


class ComponentCropper:
    """Utility for cropping components from circuit board images."""
    
    def __init__(self, padding: int = 10):
        """
        Initialize component cropper.
        
        Args:
            padding: Padding to add around cropped components (in pixels)
        """
        self.padding = padding
    
    def crop_component(
        self,
        image: np.ndarray,
        bbox: List[float],
        padding: Optional[int] = None
    ) -> np.ndarray:
        """
        Crop a single component from an image.
        
        Args:
            image: Source image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Optional padding override
            
        Returns:
            Cropped component image
        """
        padding = padding if padding is not None else self.padding
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Extract and expand bbox
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding and clip to image boundaries
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop component
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def crop_from_detections(
        self,
        image_path: str,
        detections: List[Dict],
        output_dir: str = "outputs/cropped_components",
        save_metadata: bool = True,
        component_filter: Optional[List[str]] = None
    ) -> List[str]:
        """
        Crop all detected components from an image.
        
        Args:
            image_path: Path to source image
            detections: List of detection dictionaries from detector
            output_dir: Directory to save cropped components
            save_metadata: Whether to save metadata JSON
            component_filter: List of component types to crop (e.g., ['IC']). If None, crop all.
            
        Returns:
            List of paths to saved component images
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base name for output files
        base_name = Path(image_path).stem
        
        saved_paths = []
        metadata = []
        
        # Process each detection
        for i, detection in enumerate(detections):
            # Extract information
            class_name = detection['class_name']
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Apply filter if specified
            if component_filter and class_name not in component_filter:
                continue
            
            # Crop component
            cropped = self.crop_component(image, bbox)
            
            # Generate output filename
            output_filename = f"{base_name}_{class_name}_{i}.jpg"
            output_path = output_dir / output_filename
            
            # Save cropped image
            cv2.imwrite(str(output_path), cropped)
            saved_paths.append(str(output_path))
            
            # Store metadata
            metadata.append({
                'original_image': str(image_path),
                'cropped_image': str(output_path),
                'component_type': class_name,
                'confidence': confidence,
                'bbox': bbox,
                'crop_index': i
            })
        
        # Save metadata if requested
        if save_metadata and metadata:
            metadata_path = output_dir / f"{base_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return saved_paths
    
    def crop_from_detection_file(
        self,
        detection_file: str,
        output_dir: str = "outputs/cropped_components",
        component_filter: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Crop components from multiple images using a detection results file.
        
        Args:
            detection_file: Path to detections.json file
            output_dir: Directory to save cropped components
            component_filter: List of component types to crop (e.g., ['IC']). If None, crop all.
            
        Returns:
            Dictionary mapping image paths to lists of cropped component paths
        """
        # Load detections
        with open(detection_file, 'r') as f:
            all_detections = json.load(f)
        
        all_cropped = {}
        
        # Process each image
        for image_path, detections in all_detections.items():
            print(f"\nCropping components from: {Path(image_path).name}")
            if component_filter:
                print(f"  Filter: {', '.join(component_filter)}")
            print(f"  Found {len(detections)} components")
            
            try:
                cropped_paths = self.crop_from_detections(
                    image_path,
                    detections,
                    output_dir,
                    component_filter=component_filter
                )
                all_cropped[image_path] = cropped_paths
                print(f"  Saved {len(cropped_paths)} cropped images")
            except Exception as e:
                print(f"  Error cropping from {image_path}: {e}")
        
        return all_cropped


def main():
    parser = argparse.ArgumentParser(description="Crop detected components from images")
    parser.add_argument(
        "--detection-file",
        type=str,
        help="Path to detections.json file"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image"
    )
    parser.add_argument(
        "--detections-json",
        type=str,
        help="Path to JSON file with detections for single image"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/cropped_components",
        help="Directory to save cropped components"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding around cropped components (pixels)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs='+',
        help="Component types to crop (e.g., IC LED). If not specified, crops all components."
    )
    
    args = parser.parse_args()
    
    # Initialize cropper
    cropper = ComponentCropper(padding=args.padding)
    
    # Process based on input type
    if args.detection_file:
        # Crop from detection file (multiple images)
        cropper.crop_from_detection_file(
            args.detection_file,
            args.output_dir,
            component_filter=args.filter
        )
        print(f"\nCropped components saved to: {args.output_dir}")
        
    elif args.image and args.detections_json:
        # Crop from single image
        with open(args.detections_json, 'r') as f:
            detections = json.load(f)
        
        cropped_paths = cropper.crop_from_detections(
            args.image,
            detections,
            args.output_dir,
            component_filter=args.filter
        )
        print(f"\nCropped {len(cropped_paths)} components")
        print(f"Saved to: {args.output_dir}")
        
    else:
        parser.error(
            "Either --detection-file OR both --image and --detections-json must be specified"
        )


if __name__ == "__main__":
    main()
