
from controller import Supervisor
import os
import json

CAMERA_WIDTH = 336
CAMERA_PROJECTION = "spherical"
CAMERA_FOV = 3.14159   # 180° horizontal
COMM_CHANNEL = 1 
CONTROLLER_NAME = "RobotController"  
ROTATION_OFFSET = -1.5708 


class sceneRandomizer:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.root = self.supervisor.getRoot()
        self.children = self.root.getField("children")
        self.objects = []
        self.placed_objects = []  
        self.init_objects()

        self.MIN_SENSOR_DISTANCE = 0.5 
        self.MIN_OBSTACLE_DISTANCE = 0.3 
        self.SENSOR_HEIGHT_OFFSET = 0.05  
   
   
    def init_objects(self):
        for i in range(self.children.getCount()):
            node = self.children.getMFNode(i)
            if node.getTypeName() == "Solid":
                self.objects.append(node)
            print(f"Found object: {node.getDef()}")
        return self.objects
    
    def calculate_distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
    
    def check_collision(self, position, size, object_type="sensor"):
        for placed_obj in self.placed_objects:
            placed_pos = placed_obj['position']
            placed_size = placed_obj['size']
            placed_type = placed_obj['type']
            

            distance = self.calculate_distance(position, placed_pos)
            if object_type == "robot" and placed_type == "robot":
                min_distance = self.MIN_SENSOR_DISTANCE
            elif object_type in ["sensor", "robot"] or placed_type in ["sensor", "robot"]:
                min_distance = self.MIN_OBSTACLE_DISTANCE
            else:
                min_distance = self.MIN_OBSTACLE_DISTANCE

            obj_radius = max(size) * 0.5
            placed_radius = max(placed_size) * 0.5
            required_distance = obj_radius + placed_radius + min_distance
            
            if distance < required_distance:
                return True  
        
        return False 
    
    def adjust_position_for_collision(self, original_position, size, object_type="robot", max_attempts=10):
        position = list(original_position)
        if object_type == "target":
            return position
        
        for attempt in range(max_attempts):
            if not self.check_collision(position, size, object_type):
                return position  
 
            import random
            position[0] += random.uniform(-0.2, 0.2)
            position[1] += random.uniform(-0.2, 0.2)
            position[2] = max(position[2], size[2] * 0.5 + 0.01)  
        print(f"Warning: Could not find collision-free position for {object_type} at {original_position}")
        return original_position

    def read_meta(self, scene_id):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        meta_file = os.path.join(project_root, "dataset_360fov", "dataset", "metadata", f"{scene_id}.json")
        if not os.path.exists(meta_file):
            print(f"Metadata file {meta_file} not found.")
            print(f"  Searched at: {meta_file}")
            print(f"  Project root detected as: {project_root}")
            return None
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        # print(f"Loaded metadata for scene {scene_id}: {metadata}")
    
        return metadata

    def extract_obstacles(self, scene_id):
        metadata = self.read_meta(scene_id)
        obstacles_metadata = metadata.get("inner_obstacles", [])
        # print(f"Obstacles metadata: {obstacles_metadata}")
        obstacles = {}
        for idx, obstacle in enumerate(obstacles_metadata):
            t = obstacle.get("t")
            r = obstacle.get("r")
            size = obstacle.get("size")
            obstacles[idx] = {"t": t, "r": r, "size": size}
        return obstacles
        # print(obstacles)

    def extract_targets(self, scene_id):
        metadata = self.read_meta(scene_id)
        targets_metadata = metadata.get("targets", [])
        # print(f"Targets metadata: {targets_metadata}")
        targets = {}
        for idx, target in enumerate(targets_metadata):
            t = target.get("t")
            r = target.get("r")
            size = target.get("size")
            targets[idx] = {"t": t, "r": r, "size": size}
        return targets
        # print(targets)


    def extract_robot(self, scene_id):
        metadata = self.read_meta(scene_id)
        robots_metadata = metadata.get("robots", [])
        robots = {}
        for idx, robot in enumerate(robots_metadata):
            t = robot.get("t", [0, 0, 0])
            r = robot.get("r", [0, 0, 0])
            size = robot.get("size", [0.1, 0.1, 0.1])
            robots[idx] = {"t": t, "r": r, "size": size}
        return robots
        # print(robot)

    def extract_walls(self, scene_id):
        metadata = self.read_meta(scene_id)
        walls_metadata = metadata.get("border_obstacles", [])
        # print(f"Walls metadata: {walls_metadata}")
        walls = {}
        for idx, wall in enumerate(walls_metadata):
            t = wall.get("t")
            r = wall.get("r")
            size = wall.get("size")
            walls[idx] = {"t": t, "r": r, "size": size}
        return walls
        # print(walls)






    def create_box_obstacle(self, name, translation, rotation, size):
        root = self.supervisor.getRoot()
        children_field = root.getField("children")

        children_field.importMFNodeFromString(-1, f'''
            DEF {name} Solid {{
                translation {translation[0]} {translation[1]} {translation[2]}
                rotation {rotation[0]} {rotation[1]} {rotation[2]} {rotation[3]}
                recognitionColors [1 0 0]
                children [
                    Shape {{
                        appearance PBRAppearance {{
                            baseColor 0.8 0.4 0.2
                            roughness 1
                            metalness 0
                        }}
                        geometry Box {{
                            size {size[0]} {size[1]} {size[2]}
                        }}
                    }}
                ]
                boundingObject Box {{
                    size {size[0]} {size[1]} {size[2]}
                }}
            }}
        ''')
        print(f"Created obstacle: {name} at {translation}")
        self.placed_objects.append({
            'position': translation,
            'size': size,
            'type': 'obstacle'
        })
        

    def create_marker(self, name, translation, size, color=[1, 0, 0], add_camera=False, rotation=None):

        root = self.supervisor.getRoot()
        children_field = root.getField("children")
        if size[2] == 0 or size[2] < 0.01:
            radius = size[0] / 2
            height = 0.05 
            geometry_str = f"Cylinder {{ radius {radius} height {height} }}"
        else:
            radius = size[0] / 2
            geometry_str = f"Sphere {{ radius {radius} }}"
    
        adjusted_translation = self.adjust_position_for_collision(translation, size, "target")
        adjusted_translation[2] += size[2] / 2  


        recognition_color = [0, 1, 0]  # Green for targets
        children_field.importMFNodeFromString(-1, f'''
            DEF {name} Solid {{
                translation {adjusted_translation[0]} {adjusted_translation[1]} {adjusted_translation[2]}
                recognitionColors [{recognition_color[0]} {recognition_color[1]} {recognition_color[2]}]
                children [
                    Transform {{
                        translation 0 0 0.025
                        children [
                            Shape {{
                                appearance PBRAppearance {{
                                    baseColor {color[0]} {color[1]} {color[2]}
                                    roughness 0.5
                                    metalness 0
                                    emissiveColor {color[0]*0.3} {color[1]*0.3} {color[2]*0.3}
                                }}
                                geometry {geometry_str}
                            }}
                        ]
                    }}
                ]
            }}
        ''')
        

        self.placed_objects.append({
            'position': adjusted_translation,
            'size': size,
            'type': 'target'
        })
        

        print(f"Created marker: {name} at {adjusted_translation} (adjusted from {translation})")


    def clear_scene_objects(self):
        root = self.supervisor.getRoot()
        children_field = root.getField("children")

        prefixes_to_remove = ["OBSTACLE_", "WALL_", "TARGET_", "SENSOR_", "ROBOT_"]

        for i in range(children_field.getCount() - 1, -1, -1):
            node = children_field.getMFNode(i)
            if node:
                def_name = node.getDef()
                if def_name and any(def_name.startswith(prefix) for prefix in prefixes_to_remove):
                    node.remove()
                    print(f"Removed: {def_name}")
        

        self.placed_objects = []


    def create_robot(self, name, translation, rotation, adjust=True):

        root = self.supervisor.getRoot()
        children_field = root.getField("children")
        if abs(rotation[2] - 1.0) < 1e-6:
            angle_z = rotation[3]
            camera_rotation_str = f"0 0 1 {angle_z}"
        else:
            camera_rotation_str = "0 0 1 0"

        if adjust:
            adjusted_translation = self.adjust_position_for_collision(translation, [0.35, 0.35, 0.15], "robot")
        else:
            adjusted_translation = list(translation)  # use metadata position exactly
        
        children_field.importMFNodeFromString(-1, f'''
            DEF {name} T3withCamera {{
                name "{name}"
                controller "{CONTROLLER_NAME}"
                translation {adjusted_translation[0]} {adjusted_translation[1]} {adjusted_translation[2]}
                rotation {camera_rotation_str}
                recognitionColors [0 0 1]
            }}
        ''')
        
        self.placed_objects.append({
            'position': adjusted_translation,
            'size': [0.35, 0.35, 0.15],
            'type': 'robot'
        })
        

        
        if rotation[2] == 1.0:
            print(f"Created robot: {name} at {adjusted_translation} (adjusted from {translation}) with omnidirectional camera (horizontal view, metadata Z-rotation={angle_z:.3f}) + communication")
        else:
            print(f"Created robot: {name} at {adjusted_translation} (adjusted from {translation}) with omnidirectional camera (horizontal view) + communication")


    def generate_scene(self, scene_id, max_robots=None):
        print(f"\n{'='*50}")
        print(f"Generating scene: {scene_id}")
        print(f"{'='*50}\n")
        self.clear_scene_objects()

        metadata = self.read_meta(scene_id)
        if not metadata:
            print(f"Failed to load metadata for scene {scene_id}")
            return False
        border_obstacles = metadata.get("border_obstacles", [])
        print(f"\nCreating {len(border_obstacles)} border obstacles...")
        for idx, wall in enumerate(border_obstacles):
            name = f"WALL_{idx:03d}"
            self.create_box_obstacle(
                name=name,
                translation=wall["t"],
                rotation=wall["r"],
                size=wall["size"]
            )
        
   

        inner_obstacles = metadata.get("inner_obstacles", [])
        print(f"\nCreating {len(inner_obstacles)} inner obstacles...")
        for idx, obstacle in enumerate(inner_obstacles):
            name = f"OBSTACLE_{idx:03d}"
            self.create_box_obstacle(
                name=name,
                translation=obstacle["t"],
                rotation=obstacle["r"],
                size=obstacle["size"]
            )
        

        

        targets = metadata.get("targets", [])
        print(f"\nCreating {len(targets)} target markers...")
        for idx, target in enumerate(targets):
            name = f"TARGET_{idx:03d}"
            self.create_marker(
                name=name,
                translation=target["t"],
                size=target["size"],
                color=[0.0, 1.0, 0.2]  # Green
            )
        

        robots = metadata.get("robots", [])
        if max_robots is not None:
            robots = robots[:max_robots]
        print(f"\nCreating {len(robots)} robot node(s) with cameras...")
        for idx, robot in enumerate(robots):
            name = f"ROBOT_{idx:03d}"
            self.create_robot(
                name=name,
                translation=robot["t"],
                rotation=robot["r"],
                adjust=False  # metadata positions are already valid; skip spherical approx
            )
        
        print(f"\n{'='*50}")
        print(f"Scene {scene_id} generation complete!")
        print(f"{'='*50}\n")
        
        return True


    def extract_all(self, scene_id):
        obstacles = self.extract_obstacles(scene_id)
        targets = self.extract_targets(scene_id)
        robot = self.extract_robot(scene_id)
        walls = self.extract_walls(scene_id)
        # sensors = self.extract_sensors(scene_id)
        return {
            "obstacles": obstacles,
            "targets": targets,
            "robot": robot,
            "walls": walls
        }





if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Project root detected as: {project_root}")