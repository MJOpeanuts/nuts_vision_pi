#!/usr/bin/env python3
"""
Visualization Utilities for Component Detection
Creates visualizations of detection results, statistics, and OCR outputs.
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import json


class DetectionVisualizer:
    """Visualization tools for component detection results."""
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_detection_statistics(
        self,
        detection_file: str,
        save_path: Optional[str] = None
    ):
        """
        Create visualization of detection statistics.
        
        Args:
            detection_file: Path to detections.json
            save_path: Optional path to save plot
        """
        # Load detections
        with open(detection_file, 'r') as f:
            all_detections = json.load(f)
        
        # Collect statistics
        component_counts = {}
        confidence_scores = {}
        
        for image_path, detections in all_detections.items():
            for det in detections:
                class_name = det['class_name']
                confidence = det['confidence']
                
                # Count components
                component_counts[class_name] = component_counts.get(class_name, 0) + 1
                
                # Store confidences
                if class_name not in confidence_scores:
                    confidence_scores[class_name] = []
                confidence_scores[class_name].append(confidence)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Component counts
        ax = axes[0, 0]
        classes = sorted(component_counts.keys())
        counts = [component_counts[c] for c in classes]
        ax.barh(classes, counts, color='steelblue')
        ax.set_xlabel('Count')
        ax.set_title('Component Detection Counts')
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Confidence distribution
        ax = axes[0, 1]
        all_confidences = []
        for confs in confidence_scores.values():
            all_confidences.extend(confs)
        ax.hist(all_confidences, bins=20, color='coral', edgecolor='black')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Score Distribution')
        ax.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_confidences):.2f}')
        ax.legend()
        
        # 3. Average confidence by component type
        ax = axes[1, 0]
        avg_confidences = {k: np.mean(v) for k, v in confidence_scores.items()}
        classes = sorted(avg_confidences.keys())
        avg_confs = [avg_confidences[c] for c in classes]
        ax.barh(classes, avg_confs, color='mediumseagreen')
        ax.set_xlabel('Average Confidence')
        ax.set_title('Average Confidence by Component Type')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # 4. Detections per image
        ax = axes[1, 1]
        detections_per_image = [len(dets) for dets in all_detections.values()]
        ax.hist(detections_per_image, bins=15, color='mediumpurple', edgecolor='black')
        ax.set_xlabel('Components per Image')
        ax.set_ylabel('Number of Images')
        ax.set_title('Component Distribution Across Images')
        ax.axvline(np.mean(detections_per_image), color='red', linestyle='--',
                   label=f'Mean: {np.mean(detections_per_image):.1f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Save or show
        if save_path is None:
            save_path = self.output_dir / "detection_statistics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved detection statistics to: {save_path}")
        plt.close()
    
    def plot_ocr_results(
        self,
        ocr_csv: str,
        save_path: Optional[str] = None
    ):
        """
        Visualize OCR extraction results.
        
        Args:
            ocr_csv: Path to OCR results CSV
            save_path: Optional path to save plot
        """
        # Load OCR results
        df = pd.read_csv(ocr_csv)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Success rate by component type
        ax = axes[0]
        success_by_type = df.groupby('component_type').apply(
            lambda x: (x['mpn'].notna() & (x['mpn'] != '')).sum() / len(x) * 100
        ).sort_values(ascending=True)
        
        ax.barh(success_by_type.index, success_by_type.values, color='teal')
        ax.set_xlabel('Success Rate (%)')
        ax.set_title('OCR Success Rate by Component Type')
        ax.grid(axis='x', alpha=0.3)
        
        # 2. MPN length distribution
        ax = axes[1]
        mpn_lengths = df[df['mpn'].notna() & (df['mpn'] != '')]['mpn'].str.len()
        ax.hist(mpn_lengths, bins=20, color='darkorange', edgecolor='black')
        ax.set_xlabel('MPN Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title('Extracted MPN Length Distribution')
        
        plt.tight_layout()
        
        # Save or show
        if save_path is None:
            save_path = self.output_dir / "ocr_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OCR results visualization to: {save_path}")
        plt.close()
    
    def create_annotated_grid(
        self,
        image_paths: List[str],
        detections_dict: Dict,
        grid_size: tuple = (3, 3),
        save_path: Optional[str] = None
    ):
        """
        Create a grid of annotated images.
        
        Args:
            image_paths: List of image paths
            detections_dict: Dictionary of detections
            grid_size: Grid dimensions (rows, cols)
            save_path: Optional path to save grid
        """
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        for idx, ax in enumerate(axes):
            if idx < len(image_paths):
                img_path = image_paths[idx]
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Draw detections
                detections = detections_dict.get(img_path, [])
                for det in detections:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Add label
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    cv2.putText(image, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                ax.imshow(image)
                ax.set_title(f"{Path(img_path).name}\n{len(detections)} components")
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if save_path is None:
            save_path = self.output_dir / "detection_grid.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved detection grid to: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize detection and OCR results")
    parser.add_argument(
        "--detection-file",
        type=str,
        help="Path to detections.json file"
    )
    parser.add_argument(
        "--ocr-csv",
        type=str,
        help="Path to OCR results CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Directory to save visualizations"
    )
    
    args = parser.parse_args()
    
    # Initialize visualizer
    viz = DetectionVisualizer(output_dir=args.output_dir)
    
    # Create visualizations
    if args.detection_file:
        viz.plot_detection_statistics(args.detection_file)
    
    if args.ocr_csv:
        viz.plot_ocr_results(args.ocr_csv)
    
    if not args.detection_file and not args.ocr_csv:
        print("Please provide --detection-file and/or --ocr-csv")


if __name__ == "__main__":
    main()
