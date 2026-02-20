import bpy
import numpy as np
import sys
import argparse
import os

def create_armature(joints, output_path):
    # joints shape: [nframes, njoints, 3]
    nframes, njoints, _ = joints.shape

    # SMPL Joint Hierarchy (Parent indices)
    # -1 means it's the root
    parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
    
    # Create Armature
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    arm_obj = bpy.context.object
    arm_data = arm_obj.data
    arm_data.name = "MDM_Armature"
    
    # Remove the default bone
    bpy.ops.armature.bone_primitive_add() # Add one just in case
    bones = arm_data.edit_bones
    for b in bones:
        bones.remove(b)

    # Create bones based on the first frame (T-pose or initial pose)
    joint_positions = joints[0]
    for i in range(njoints):
        bone = bones.new(f"joint_{i}")
        bone.head = joint_positions[i]
        # End of bone is just slightly offset or towards parent
        bone.tail = joint_positions[i] + np.array([0, 0, 0.1])
        
        if parents[i] != -1:
            bone.parent = bones[f"joint_{parents[i]}"]
            bone.use_connect = False # Keep positions as generated

    bpy.ops.object.mode_set(mode='POSE')

    # Animate bones
    for f in range(nframes):
        bpy.context.scene.frame_set(f + 1)
        for i in range(njoints):
            pbone = arm_obj.pose.bones[f"joint_{i}"]
            # MDM usually provides global coordinates
            # We set the bone location in world space
            # This is a bit simplified; a robust script would handle local rotations
            pbone.location = joints[f, i]
            pbone.keyframe_insert(data_path="location", frame=f + 1)

    # Export to FBX
    bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, add_leaf_bones=False)

if __name__ == "__main__":
    # Internal Blender args start after "--"
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args = sys.argv[idx+1:]
    else:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    opt = parser.parse_args(args)

    # Load MDM results
    data = np.load(opt.input, allow_pickle=True).item()
    # MDM npy format: dict with 'motion' key or similar
    # If it's the raw results.npy from sample.generate:
    # shape is [num_reps, num_joints,Feats, num_frames]
    
    motions = data['motion'] 
    # Let's take the first repetition
    motion = motions[0] # [njoints, 3, nframes]
    motion = motion.transpose(2, 0, 1) # [nframes, njoints, 3]

    create_armature(motion, opt.output)
