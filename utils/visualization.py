import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def draw_detections(frame, detections, activity=None, confidence=None):
    """
    Draw bounding boxes and labels on frame
    
    Args:
        frame (numpy.ndarray): Input frame
        detections (list): List of detections [x1, y1, x2, y2, confidence, class_id]
        activity (str, optional): Detected activity
        confidence (float, optional): Activity confidence score
        
    Returns:
        numpy.ndarray: Annotated frame
    """
    annotated = frame.copy()
    
    # Draw each detection
    for x1, y1, x2, y2, conf, class_id in detections:
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Person: {conf:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1])
        
        cv2.rectangle(annotated, (x1, y1_label - label_size[1]), (x1 + label_size[0], y1_label), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add activity label if provided
    if activity:
        activity_label = f"Activity: {activity}"
        if confidence:
            activity_label += f" ({confidence:.2f})"
        
        cv2.putText(annotated, activity_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, timestamp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return annotated

def create_activity_histogram(activities, title="Activity Distribution"):
    """
    Create histogram of detected activities
    
    Args:
        activities (list): List of activity dictionaries
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: Histogram figure
    """
    # Count activities
    activity_counts = {}
    for entry in activities:
        activity = entry["activity"]
        if activity in activity_counts:
            activity_counts[activity] += 1
        else:
            activity_counts[activity] = 1
    
    # Sort by count (descending)
    sorted_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [a[0] for a in sorted_activities]
    counts = [a[1] for a in sorted_activities]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(labels, counts, color='skyblue')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(count),
            ha='center',
            va='bottom'
        )
    
    # Customize chart
    ax.set_title(title)
    ax.set_xlabel("Activity")
    ax.set_ylabel("Count")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_activity_timeline(activities, title="Activity Timeline", limit=20):
    """
    Create timeline of detected activities
    
    Args:
        activities (list): List of activity dictionaries
        title (str): Plot title
        limit (int): Maximum number of activities to display
        
    Returns:
        matplotlib.figure.Figure: Timeline figure
    """
    # Limit number of activities
    activities = activities[:limit]
    
    # Extract timestamps and activity names
    timestamps = [datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S") for a in activities]
    activities_names = [a["activity"] for a in activities]
    
    # Get unique activities for color mapping
    unique_activities = list(set(activities_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_activities)))
    color_map = {activity: color for activity, color in zip(unique_activities, colors)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create timeline
    for i, (timestamp, activity) in enumerate(zip(timestamps, activities_names)):
        ax.scatter(timestamp, i, color=color_map[activity], s=100)
        ax.text(timestamp, i + 0.1, activity, ha='center', va='bottom', fontsize=8)
    
    # Customize chart
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_yticks([])
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
               for activity, color in color_map.items()]
    ax.legend(handles, color_map.keys(), title="Activities")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig