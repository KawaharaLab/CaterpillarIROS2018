import os
import re
import json
import maya.cmds as cmds

def load_json_render(file_path):
    with open(file_path, 'r') as f:
        raw_sim_proc = json.load(f)
    return {
        'objects': [(o['rad'], o['pos'], o['id']) for o in raw_sim_proc['objects']],
        'frames': {int(i): [(o['id'], tuple(o['pos'])) for o in v] for i, v in  raw_sim_proc['frames'].items()}
    }


cmds.select('Ctrl_Root')
cmds.select(hierarchy=True)
oSel = cmds.ls(sl=True)

for o_name in oSel:
    # print "cut keyframes of {}".format(o_name)
    cmds.cutKey(o_name)

base_dir = os.path.join("/Users/luning/ResearchMaster/src/iros/gripping/results/muscle/normal/k9/run0")
trials = filter(lambda x: re.match(r'trial[0-9]+', x), os.listdir(base_dir))
render_files = []
for trial in trials:
    rfile = os.path.join(base_dir, trial, "run/render.json")
    render_files.append(rfile)
    # print rfile

rf = render_files[12]
print "read {}".format(rf)
render_json_obj = load_json_render(rf)

objs = render_json_obj['objects']
frames = render_json_obj['frames']

somites_bones = {
    '_somite_0': [1],
    '_somite_1': [3],
    '_somite_2': [5],
    '_somite_3': [7],
    '_somite_4': [9],
}

# # update positions
for keyframe in range(len(frames)):
    prev_somite_pos = None
    # frames[keyframe].reverse()
    for s_id, pos in frames[keyframe]:
        if s_id == '_somite_0':
            for bone_id in somites_bones[s_id]:
                cmds.setKeyframe('Bone{0:03d}'.format(bone_id), t=keyframe ,v=-8*pos[0], at='translateZ')
                cmds.setKeyframe('Bone{0:03d}'.format(bone_id), t=keyframe ,v=pos[1], at='translateX')
                cmds.setKeyframe('Bone{0:03d}'.format(bone_id), t=keyframe ,v=8*pos[2], at='translateY')
        else:
            for bone_id in somites_bones[s_id]:
                cmds.setKeyframe('Bone{0:03d}'.format(bone_id), t=keyframe ,v=-8*(pos[0] - prev_somite_pos[0]), at='translateZ')
                cmds.setKeyframe('Bone{0:03d}'.format(bone_id), t=keyframe ,v=pos[1] - prev_somite_pos[1], at='translateX')
                cmds.setKeyframe('Bone{0:03d}'.format(bone_id), t=keyframe ,v=8*(pos[2] - prev_somite_pos[2]), at='translateY')

        prev_somite_pos = (pos[0], pos[1], pos[2])
