import pandas as pd
import rp

pano_vids = [
    # Hand-selected for minimal camera movement and maximal coolness
    "WEB360/videos_512x1024x100/100251.mp4",
    "WEB360/videos_512x1024x100/100283.mp4",
    "WEB360/videos_512x1024x100/100296.mp4",
    "WEB360/videos_512x1024x100/100300.mp4",
    "WEB360/videos_512x1024x100/100297.mp4",
    "WEB360/videos_512x1024x100/100335.mp4",
    "WEB360/videos_512x1024x100/100355.mp4",
    "WEB360/videos_512x1024x100/100360.mp4",
    "WEB360/videos_512x1024x100/100359.mp4",
    "WEB360/videos_512x1024x100/100384.mp4",
    "WEB360/videos_512x1024x100/100405.mp4",
    "WEB360/videos_512x1024x100/100440.mp4",
    "WEB360/videos_512x1024x100/100481.mp4",
    "WEB360/videos_512x1024x100/100488.mp4",
    "WEB360/videos_512x1024x100/100487.mp4",
    "WEB360/videos_512x1024x100/100485.mp4",
    "WEB360/videos_512x1024x100/100489.mp4",
    "WEB360/videos_512x1024x100/100492.mp4",
    "WEB360/videos_512x1024x100/100524.mp4",
    "WEB360/videos_512x1024x100/100531.mp4",
    "WEB360/videos_512x1024x100/100575.mp4",
    "WEB360/videos_512x1024x100/100604.mp4",
    "WEB360/videos_512x1024x100/100602.mp4",
    "WEB360/videos_512x1024x100/100640.mp4",
    "WEB360/videos_512x1024x100/100642.mp4",
    "WEB360/videos_512x1024x100/100675.mp4",
    "WEB360/videos_512x1024x100/100712.mp4",
    "WEB360/videos_512x1024x100/100726.mp4",
    "WEB360/videos_512x1024x100/100737.mp4",
    "WEB360/videos_512x1024x100/100773.mp4",
    "WEB360/videos_512x1024x100/100783.mp4",
    "WEB360/videos_512x1024x100/100785.mp4",
    "WEB360/videos_512x1024x100/100791.mp4",
    "WEB360/videos_512x1024x100/100810.mp4",
    "WEB360/videos_512x1024x100/100817.mp4",
    "WEB360/videos_512x1024x100/100941.mp4",
    "WEB360/videos_512x1024x100/101015.mp4",
    "WEB360/videos_512x1024x100/101065.mp4",
    "WEB360/videos_512x1024x100/101137.mp4",
    "WEB360/videos_512x1024x100/101138.mp4",
    "WEB360/videos_512x1024x100/101139.mp4",
    "WEB360/videos_512x1024x100/101136.mp4",
    "WEB360/videos_512x1024x100/101170.mp4",
    "WEB360/videos_512x1024x100/101192.mp4",
    "WEB360/videos_512x1024x100/101656.mp4",
    "WEB360/videos_512x1024x100/101719.mp4",
    "WEB360/videos_512x1024x100/101714.mp4",
    "WEB360/videos_512x1024x100/101715.mp4",
    "WEB360/videos_512x1024x100/101717.mp4",
]

csv = pd.read_csv("WEB360/WEB360_360TF_train.csv")
id_to_caption = {videoid: name for videoid, name in zip(csv.videoid, csv.name)}
videoids = [rp.get_file_name(file, include_file_extension=False) for file in pano_vids]

captions = [id_to_caption[int(videoid)] for videoid in videoids]

pano_vids, captions = rp.sync_shuffled(pano_vids, captions)
