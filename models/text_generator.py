import random
import logging

logger = logging.getLogger(__name__)

class TextGenerator:
    """
    Generate text descriptions of detected activities
    """
    def __init__(self, template_based=True, templates=None):
        """
        Initialize text generator
        
        Args:
            template_based (bool): Whether to use template-based generation
            templates (list): List of templates for generation
        """
        self.template_based = template_based
        
        # Default templates if none provided
        self.templates = templates if templates else [
            "A person is {activity} in the surveillance area.",
            "The camera detected a person {activity}.",
            "Surveillance shows a human {activity}.",
            "The system identified a person {activity}.",
            "A human was observed {activity} in the monitored area."
        ]
        
        # Advanced template mapping to improve grammar
        self.activity_template_map = {
            "walking": [
                "walking in the surveillance area",
                "walking across the monitored zone",
                "moving through the area"
            ],
            "running": [
                "running in the surveillance area",
                "running across the monitored zone",
                "moving quickly through the area"
            ],
            "standing": [
                "standing in the surveillance area",
                "standing still in the monitored zone",
                "remaining stationary in the area"
            ],
            "sitting": [
                "sitting in the surveillance area",
                "seated in the monitored zone",
                "resting in the area"
            ],
            "bending": [
                "bending over in the surveillance area",
                "leaning down in the monitored zone",
                "stooping in the area"
            ],
            "falling": [
                "falling in the surveillance area",
                "collapsing in the monitored zone",
                "losing balance in the area"
            ],
            "lying": [
                "lying down in the surveillance area",
                "lying on the ground in the monitored zone",
                "resting horizontally in the area"
            ],
            "crawling": [
                "crawling in the surveillance area",
                "moving on hands and knees in the monitored zone",
                "crawling across the area"
            ],
            "fighting": [
                "involved in a physical altercation",
                "engaged in a fight",
                "physically conflicting with others"
            ],
            "waving": [
                "waving in the surveillance area",
                "gesturing with their hand in the monitored zone",
                "signaling in the area"
            ],
            "using_phone": [
                "using a phone in the surveillance area",
                "on a mobile device in the monitored zone",
                "engaged with a smartphone in the area"
            ],
            "other": [
                "performing an unclassified activity",
                "engaged in an unrecognized action",
                "doing something in the monitored area"
            ],
            "unknown": [
                "present in the surveillance area",
                "detected in the monitored zone",
                "visible in the area"
            ]
        }
    
    def generate(self, activity, confidence=None):
        """
        Generate a text description of the detected activity
        
        Args:
            activity (str): Detected activity
            confidence (float, optional): Confidence score
            
        Returns:
            str: Generated text description
        """
        if self.template_based:
            # Select template randomly
            template = random.choice(self.templates)
            
            # Get activity description with improved grammar
            if activity in self.activity_template_map:
                activity_desc = random.choice(self.activity_template_map[activity])
            else:
                activity_desc = activity
            
            # Fill template with activity
            description = template.format(activity=activity_desc)
            
            # Add confidence if provided
            if confidence is not None:
                description += f" (Confidence: {confidence:.2f})"
            
            return description
        else:
            # For future implementation of more advanced text generation
            # This could be expanded to use a language model for more natural descriptions
            logger.warning("Advanced text generation not implemented yet, falling back to template-based")
            return self.generate(activity, confidence)
