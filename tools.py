import collections
import numpy as np
import pygame
import torch
import time

import world # in the file world.py, constants like gravity strength are set 

# parse an animal from a configuration file
def parse_animal(file):
        
    mode = None # defines which section in the file is read at the moment
    nodes = collections.OrderedDict() # the points where bones and muscles are attached
    bones = collections.OrderedDict() # bones connecting two nodes
    muscles = collections.OrderedDict() # muscles connecting two nodes 
    bone_connects = collections.OrderedDict()
    muscle_connects = collections.OrderedDict()
    coords = collections.OrderedDict()
    
    with open(file) as f:
        for line in f.readlines(): # read all lines of the file
            if mode == "NODES": # if we are at the "NODES" section of the file
                if(len(line.split()) == 4): # check if there are 4 columns
                    split = line.split() # split the 4 columns to a list
                    nodes[split[0]] = float(split[1]) # the node with the name split[0] has mass split[1]
                    coords[split[0]] = [float(split[2]),float(split[3])] # the node with the name split[0] starts at coordinates x=float(split[2]), y=float(split[3])
            if mode == "BONES": # if we are at the "BONES" section of the file
                if(len(line.split()) == 4): # check if there are 4 columns
                    split = line.split() # split the 4 columns to a list
                    bones[(split[0],split[1])] = (float(split[2]),float(split[3])) # the bone connecting node split[0] to split[1] has force constant split[2] and equilibrium length split[3]
                    bone_connects[split[0]].add(split[1]) # split[0] is connected via a bone to split[1]
                    bone_connects[split[1]].add(split[0]) # vice versa, split[1] is connected via a bone to split[0]
            if mode == "MUSCLES": # if we are at the "MUSCLES" section of the file
                if(len(line.split()) == 6): # check if there are 6 columns
                    split = line.split() # split the 6 columns to a list
                    muscles[(split[0],split[1])] = (float(split[2]),float(split[3]),float(split[4]),float(split[5])) # the muscle connecting node split[0] to split[1] 
                    # has force constant split[2] and equilibrium length split[3] when flexed and force constant split[4] and equilibrium length split[5] when tensed    
                    muscle_connects[split[0]].add(split[1]) # split[0] is connected via a muscle to split[1]
                    muscle_connects[split[1]].add(split[0]) # vice versa, split[1] is connected via a muscle to split[0]
            
            if line.strip() == "NODES": # check if we are entering the "NODES" section
                mode = "NODES" # if yes, set the mode accordingly  
 
            if line.strip() == "BONES": # check if we are entering the "BONES" section
                mode = "BONES" # if yes, set the mode accordingly   
                for i in nodes.keys():
                    bone_connects[i] = set() # initialize the bone connects for every node as an empty set

            if line.strip() == "MUSCLES": # check if we are entering the "MUSCLES" section
                mode = "MUSCLES" # if yes, set the mode accordingly
                for i in nodes.keys():
                    muscle_connects[i] = set() # initialize the muscle connects for every node as an empty set

    return nodes, bones, muscles, bone_connects, muscle_connects, coords # return the filled arrays
 
# put the center of a set of coordinates to the middle of the screen    
def center(coords):
    
    # find the center of a set of coordinates     
    x_mean = 0
    y_mean = 0
    for i in coords.values():
        x_mean += i[0]
        y_mean += i[1]
    x_mean /= len(coords)
    y_mean /= len(coords)
    
    # shift the center of mass to the middle of the screen (coordinates units are in percent of the screen)
    for i in coords.values(): 
        i[0] += 50 - x_mean # center to 50 percent
        i[1] += 50 - y_mean
        
    return coords
    
# calculate the potential energy in the animal due to muscles/bones not residing in their equilibrium position
def energy(device, coords, bones, bones_indices, muscles, flexes, muscles_indices):
    
    energy = torch.zeros(1,dtype=torch.float32).to(device) # initialize the enegry as 0 and load it to the device
    
    i = 0 # initialize a counter to zero
    dists = torch.zeros(len(bones), dtype=torch.float32).to(device) # alloctate an array for the extensions of the bones
    for b in bones_indices: # for all bones
        tmp = coords[b[1]]-coords[b[0]] # get the vector pointing from node b[0] to b[1]
        dists[i] = torch.sqrt( torch.dot(tmp,tmp) ) # get the extension as the length of the tmp vector and set it as dists[i]
        i += 1 # increase the counter

    # get the total energy of the bones as a sum of the Hook's law [E = (x - x0)**2 * K, x0 == equilibrium length, k == force constant] energies of all bones
    energy += torch.sum( (dists-bones[:,1])**2 * bones[:,0]) 
    
    i = 0 # initialize the counter to zero
    dists = torch.zeros(len(muscles)).to(device) # alloctate an array for the extensions of the muscles
    for m in muscles_indices: # for all muscles
        tmp = coords[m[1]]-coords[m[0]] # get the vector pointing from node b[0] to b[1]
        dists[i] = torch.sqrt( torch.dot(tmp,tmp) ) # get the extension as the length of the tmp vector and set it as dists[i]
        i+=1 # increase the counter
    
    # get the total energy of the bones as a sum of the Hook's law [E = (x - x0)**2 * K/2, x0 == equilibrium length, K/2 == force constant] energies of all muscles
    # flexes[:,0] + flexes[:,1] == 1 specifies the amount of muscle tension: the first term in the sum gives the energy due to the flexed amount, the second due to the tensed amount
    energy += torch.sum( (dists-muscles[:,1])**2 *muscles[:,0] * flexes[:,0] + (dists-muscles[:,3])**2 * muscles[:,2] * flexes[:,1])
    
    return energy # return the energy

# minimize the energy of the animal by using the the adam machine learning optimizer with the energy as the cost function    
def minimize_energy(device, coords, bones, bones_indices, muscles, flexes, muscles_indices):
    
    optimizer = torch.optim.Adam(params=[coords], lr=world.LEARNING_RATE_MINIM, betas=(0.5, 0.999)) # initialize the Adam optimizer for the coordinates
    
    for i in range(world.EPOCHS_MINIM): # do world.EPOCHS_MINIM (imported from world.py) minimization steps
        optimizer.zero_grad() # set the gradients to zero
        ener = energy(device, coords, bones, bones_indices, muscles, flexes, muscles_indices) # calculate the cost function
        if( i % (world.EPOCHS_MINIM / 10) == 0 ): print("Minimizer Epoch: "+str(i)+" Energy: "+ str(ener)) # print statistics 
        ener.backward() # backpropagate
        optimizer.step() # perform an optimization step on the coordinates


# calculate the forces and apply them to accelarate the velocities. This function will be rewritten (TODO) in matrix form soon, which is why it is loosely documented        
def apply_accelerations(device, coords, bones, bones_indices, muscles, flexes, muscles_indices, masses, velocities, timestep):
    vels = velocities
    
    i = 0 # accelarate due to forces from the bones
    for b in bones_indices:
        tmp = coords[b[1]]-coords[b[0]]
        dist = torch.sqrt(torch.dot(tmp,tmp))
        uvec = tmp / dist
        force = (dist-bones[i,1])*bones[i,0]
        vels[b[0]] += force * uvec / masses[b[0]] * timestep
        vels[b[1]] -= force * uvec / masses[b[1]] * timestep   
        i += 1
        
    i = 0 # accelarate due to forces from the muscles
    for b in muscles_indices:
        tmp = coords[b[1]]-coords[b[0]]
        dist = torch.sqrt(torch.dot(tmp,tmp))
        uvec = tmp / dist
        force = (dist-muscles[i,1])*muscles[i,0]*flexes[i,0] + (dist-muscles[i,3])*muscles[i,2]*flexes[i,1]
        vels[b[0]] += force * uvec  / masses[b[0]] * timestep
        vels[b[1]] -= force * uvec  / masses[b[1]] * timestep   
        i += 1
    
    # accelarte due to gravitational force
    vels += torch.tensor([[0,-world.GRAVITY_CONSTANT * timestep] for i in vels],dtype=torch.float32).to(device)
    
    # accelerate due to hitting the ground    
    for i in range(len(coords)):
        if coords[i,1] < 0:
            vels[i,1] += world.GROUND_FORCE * timestep / masses[i]
    
    # decelerate due to viscosity of the medium
    for i in range(len(vels)):
        vels[i] -= vels[i] * world.VISCOSITY * timestep / masses[i]
    
    return vels # return the modified velocities
   
# change the coordinates according to the velocities and get the new velocities in leap-frog style (inspired by molecular dynamics)    
def integrate_leapfrog(device, coords, bones, bones_indices, muscles, flexes, muscles_indices, masses, velocities, timestep):
    
    coords += velocities * timestep # integrate the coordinates
    apply_accelerations(device, coords, bones, bones_indices, muscles, flexes, muscles_indices, masses, velocities, timestep) # as well as the velocities 



# draw the animal
def draw(coords, bones, muscles):
    
    pygame.display.init() # intialize the pygame environment for drawing
    
    # define colors
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    # get a canvas and fill it with black color
    gameDisplay = pygame.display.set_mode( (world.CANVAS_X, world.CANVAS_Y) )
    gameDisplay.fill(black)
    
    # transfrom the coordinates from units of percent of the canvas to actual pixel values. Note that int(x + 0.5) == round(x).  
    coordinates = [ [ int( i[0] / 100.0 * world.CANVAS_X + 0.5), world.CANVAS_Y - int(i[1] / 100.0 * world.CANVAS_Y + 0.5) ] for i in coords.values() ]
    
    # draw a green circle for evey node
    for i in coordinates:
        pygame.draw.circle(gameDisplay, green, i, 20)

    # draw a white line for every bone
    for i in bones.keys():
        pygame.draw.line(gameDisplay, white, coordinates[ list(coords.keys()).index(i[0]) ], coordinates[ list(coords.keys()).index(i[1]) ], 5)
    
    # draw a red line for every muscle    
    for i in muscles.keys():
        pygame.draw.line(gameDisplay, red, coordinates[ list(coords.keys()).index(i[0]) ], coordinates[ list(coords.keys()).index(i[1]) ], 5)

    # keep the window open until the user closes it 
    def loop():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    return
            pygame.display.update()
    loop()    


# simulate the animal and play the clip    
def watch_simulation(device, coords, bones, bones_indices, muscles, flexes, muscles_indices, masses, velocities, timestep, nSteps):
    
    pygame.display.init() # intialize the pygame environment for drawing

    # define colors
    white = (255,255,255)
    black = (0,0,0)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)

    # get a canvas
    gameDisplay = pygame.display.set_mode( (world.CANVAS_X,world.CANVAS_Y) )
    
    # perform the simulation and draw until the user closes it
    def loop():
        # transfrom the coordinates from units of percent of the canvas to actual pixel values. Note that int(x + 0.5) == round(x).  
        coords_print = [ [int( i[0] / 100.0 * world.CANVAS_X + 0.5 ), world.CANVAS_Y - int( i[1] / 100.0 * world.CANVAS_Y + 0.5 ) ] for i in coords ]
        for step in range(nSteps): # for the specified amount of steps
            if(step % world.SKIP_FRAMES == 0): # only draw every world.SKIP_FRAMES frame
                gameDisplay.fill(black) # fille the canvas black
                for i in coords_print: # draw a green circle for evey node 
                    pygame.draw.circle(gameDisplay, green, i, 20)
                for i in bones_indices: # draw a white line for every bone
                    pygame.draw.line(gameDisplay, white, coords_print[i[0]], coords_print[i[1]],5)
                for i in muscles_indices: # draw a red line for every muscle
                    pygame.draw.line(gameDisplay, red, coords_print[i[0]], coords_print[i[1]],5)
            
            for event in pygame.event.get(): # if the user closes the window, quit pygame and return from the loop function
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    return
            pygame.display.update() # draw everything on the canvas
            
            integrate_leapfrog(device, coords, bones, bones_indices, muscles, flexes, muscles_indices, masses, velocities, timestep) # integrate to get the next coordinates and velocities
            
            # transfrom the coords_print from units of percent of the canvas to actual pixel values            
            coords_print = [ [int( i[0] / 100.0 * world.CANVAS_X + 0.5 ), world.CANVAS_Y - int( i[1] / 100.0 * world.CANVAS_Y + 0.5)] for i in coords]
            
            # apply the ai agents policy, i. e. tense/flex the muscles
            policy(coords, flexes, timestep)
    loop()
    
    
# The policy of the reinforcement learning agent. The current policy is just a random dummy policy for demonstration purposes. Here is where you want to start coding.
def policy(coords, flexes, timestep):
    for flex in flexes:
        if np.random.uniform() < world.RANDOM_TENSION_CHANGE * timestep: # check if the muscle state (i. e. flexed or tensed) gets changed at all
            if np.random.uniform() < world.RANDOM_CHANGE_TO_TENSION: # check to which state it gets changed
                flex[0]=0 # change to 0% flex 
                flex[1]=1 # and 100% tension
            else:
                flex[0]=1 # chang to 100% flex
                flex[1]=0 # and 0% tension
                
    

    
        
    
