import torch

import tools
import world


# load an animal consisting of bones and muscles
nodes, bones, muscles, bone_connects, muscle_connects, coords = tools.parse_animal("bat.txt")

# specify the device on which to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# center the coordinates of the animal (coords are in units of percent of the screen)
coords = tools.center(coords)


# initialize the torch tensors. We transform from OrderedDicts (node, bones, muscles, bone_connects, muscle_connects, coords) to tensor formalism
# see parse_animal in tools.py and animal.txt for the exact meaning

# the coordinates will be energy minimized before the simulation, so we set requires_grad=True 
coords_d = torch.tensor( list(coords.values()), dtype=torch.float32, requires_grad=True, device=device ) 

# this tensor contains the force constants and equilibrium values of the bones. Bones is an OrderedDict, so the identity is preserved by the order
bones_d = torch.tensor( [ [ v[0], v[1] ] for v in bones.values() ], dtype=torch.float32 ).to(device)

# get the corresponding incices in coords_d for the bones endpoint nodes
bones_indices_d = torch.tensor( [ [ list(coords.keys()).index(i[0]), list(coords.keys()).index(i[1]) ] for i in bones.keys() ], dtype=torch.int16 ).to(device) 

# this tensor contains the force constants and equilibrium values of the muscles in flexed and tensed state. muscles is an OrderedDict, so the identity is preserved by the order
muscles_d = torch.tensor( [ [ v[0], v[1], v[2], v[3] ] for v in muscles.values() ], dtype=torch.float32 ).to(device) 

# get the corresponding incices in coords_d for the muscle endpoint nodes
muscles_indices_d = torch.tensor( [ [ list(coords.keys()).index(i[0]), list(coords.keys()).index(i[1]) ] for i in muscles.keys() ],dtype=torch.int16 ).to(device) 

# initialize all muscles as flexed, [1,0] means 100% flexed, 0% tension 
flexes_d = torch.tensor( [ [1,0] for i in range(len(muscles)) ], dtype=torch.float32 ).to(device)

# initialize all 2D velocities as [0,0] 
velocities_d = torch.tensor( [ [0,0] for i in range(len(coords)) ], dtype=torch.float32 ).to(device)

# get a torch tensor of the masses of the nodes. masses is an OrderedDict, so the identity is preserved by the order
masses_d = torch.tensor( list(nodes.values()), dtype=torch.float32, device=device)

# Use the Adam Optimizer to find the minimum energy structure
tools.minimize_energy(device,coords_d,bones_d,bones_indices_d,muscles_d,flexes_d,muscles_indices_d)

#  if you want to draw the minimized structure, uncomment the following
# i = 0
# for k in coords.keys():
#     coords[k] = coords_d.to("cpu").tolist()[i]
#     i += 1
# tools.draw(coords, bones, muscles)

#  to perform a single leap-frog integration step, uncomment the following code
# velocities_d = tools.apply_accelerations(device, coords_d, bones_d, bones_indices_d, muscles_d, flexes_d, muscles_indices_d, masses_d, velocities_d,0.5)
# coords_d.requires_grad = False
# coords_d, velocities_d = tools.integrate_leapfrog(device, coords_d,bones_d, bones_indices_d, muscles_d, flexes_d, muscles_indices_d, masses_d, velocities_d,1)


# this finally runs the simulation
coords_d.requires_grad = False
tools.watch_simulation(device, coords_d, bones_d, bones_indices_d, muscles_d, flexes_d, muscles_indices_d, masses_d, velocities_d, 0.01, 10000)

