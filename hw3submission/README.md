All graded output images are in ./outputs (hw_3_1_1.png through hw_3_6_5.png).

The custom scene that we designed is in scenes/custom_one_ring_to_rule_them_all.json, and every mesh it uses is in scenes/custom_assets/.

Its rendering is outputs/custom_one_ring_to_rule_them_all.png.
We used our 3_6 code to render it.
Copy custom_one_ring_to_rule_them_all.json and the custom_assets folder into scenes/hw3/ and run

`uv run python main.py -output outputs/custom_one_ring_to_rule_them_all.png 3_6 scenes/hw3/custom_one_ring_to_rule_them_all.json`

The credits/explanation for the custom scene:

The main subject of the scene is a large, golden ring.
The top of the ring is the symbol of ChatGPT aka OpenAI.
It overshadows a smaller golden ring intended to represent 
the One Ring from J.R.R. Tolkien's The Lord of the Rings books.
I think you can infer the intended symbolism!

Ring band and blossom (band.ply, blossom.ply):
Made by Luke's friend Jeremy Gu (jeremygu@berkeley.edu).
Jeremy is a UC Berkeley mathematics undergrad with an
interest in computer graphics.
Shoot him a quick email if time allows!

The One Ring (the_one_ring_lord_of_the_rings.ply):
"The One Ring (Lord of The Rings)" by Yanez Designs
https://sketchfab.com/3d-models/the-one-ring-lord-of-the-rings-39eb401be92c49d39520fadd5ecff8d3
License: CC Attribution 4.0 (http://creativecommons.org/licenses/by/4.0/)
(did not use the texture)

The walls are made from the provided quad meshes. scenes/hw3/meshes/quad_mesh.json

