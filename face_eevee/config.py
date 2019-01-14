import tempfile
import os
import urllib.request
import bz2

eas_master_folder = "https://raw.githubusercontent.com/patrikhuber/eos/master/share/"

share_files  = [
    'sfm_shape_3448.bin',
    'expression_blendshapes_3448.bin',
    'ibug_to_sfm.txt',
    'sfm_3448_edge_topology.json',
    'ibug_to_sfm.txt',
    'sfm_model_contours.json',
]

share_path = os.path.join(tempfile.gettempdir(), 'eos_share')
os.makedirs(share_path, exist_ok=True)

print("Downloading pretrained model of https://github.com/patrikhuber/eos")
print("Go read licence for usage right")
for share_file in share_files:
    print("Needs ", os.path.join(share_path, share_file))
    if os.path.isfile(os.path.join(share_path, share_file)):
        print("File already exists")
    else:
        print("Downloading ", os.path.join(share_path, share_file))
        urllib.request.urlretrieve(eas_master_folder + share_file, filename=os.path.join(share_path, share_file))


url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
p = os.path.join(tempfile.gettempdir(), 'shape_predictor_68_face_landmarks.dat')
print("Downloading pretrained Dlib landmark model to ", p)

if os.path.isfile(p):
        print("File already exists")
else:
    print("Downloading ", p)
    print('unziping ', p+'.bz2')
    with open(p, 'wb') as new_file, bz2.BZ2File(p+'.bz2', 'rb') as f:
        for data in iter(lambda : f.read(100 * 1024), b''):
            new_file.write(data)
