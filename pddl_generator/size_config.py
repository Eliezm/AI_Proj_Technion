# -*- coding: utf-8 -*-
"""
Problem size configurations for each domain.
Defines what "small", "medium", "large" mean for each domain.
"""

SIZE_CONFIGS = {
    "blocks_world": {
        "small": {
            "num_blocks": 4,
            "num_stacks": 2,
            "description": "4 blocks, 2 stacks"
        },
        "medium": {
            "num_blocks": 8,
            "num_stacks": 3,
            "description": "8 blocks, 3 stacks"
        },
        "large": {
            "num_blocks": 12,
            "num_stacks": 4,
            "description": "12 blocks, 4 stacks"
        },
    },
    "logistics": {
        "small": {
            "num_cities": 2,
            "locations_per_city": 2,
            "trucks_per_city": 1,
            "objects": 4,
            "description": "2 cities, 4 objects"
        },
        "medium": {
            "num_cities": 3,
            "locations_per_city": 2,
            "trucks_per_city": 1,
            "objects": 8,
            "description": "3 cities, 8 objects"
        },
        "large": {
            "num_cities": 4,
            "locations_per_city": 3,
            "trucks_per_city": 2,
            "objects": 12,
            "description": "4 cities, 12 objects"
        },
    },
    "gripper": {
        "small": {
            "num_rooms": 3,
            "num_objects": 4,
            "num_grippers": 2,
            "description": "3 rooms, 4 objects"
        },
        "medium": {
            "num_rooms": 4,
            "num_objects": 8,
            "num_grippers": 2,
            "description": "4 rooms, 8 objects"
        },
        "large": {
            "num_rooms": 5,
            "num_objects": 12,
            "num_grippers": 2,
            "description": "5 rooms, 12 objects"
        },
    },
}