All graded output images are in ./outputs (hw_4_1_1.mp4 through hw_4_6_4.mp4).

The custom scene that we designed is in scenes/custom_scene.py.
It uses only one custom mesh which is in custom_assets.

Its rendering is outputs/custom_scene.mp4
We used 4_6 to render it.
copy custom_scene.py and the custom_assets folder into scenes/hw4/, then run

`uv run python main.py 4_6 scenes/hw4/custom_scene.py -output outputs/custom_scene.mp4 -duration 5 -fps 30`

Credits for the bear asset:
Bear model (bear-sp22.ply)
  From UCSD CSE 125: https://cse125.ucsd.edu/data/bear.html

The custom scene is some text made of string ("HELLO 167") falling on top of some bear models. Then a big cloth also falls.