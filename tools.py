import collections
import numpy as np
import pygame
import torch
import time

import world

def parse_animal(file):
        
    mode = None
    nodes = collections.OrderedDict()
    bones = collections.OrderedDict()
    muscles = collections.OrderedDict()
    bone_connects = collections.OrderedDict()
    muscle_connects = collections.OrderedDict()
    coords = collections.OrderedDict()
    
    with open(file) as f:
        for line in f.readlines():
            if mode == "NODES":
                if(len(line.split()) == 4):
                    split = line.split()
                    nodes[split[0]] = float(split[1])
                    coords[split[0]] = [float(split[2]),float(split[3])]
            if mode == "BONES":
                if(len(line.split()) == 4):
                    split = line.split()
                    bones[(split[0],split[1])] = (float(split[2]),float(split[3]))
                    bone_connects[split[0]].add(split[1])
                    bone_connects[split[1]].add(split[0])
            if mode == "MUSCLES":
                if(len(line.split()) == 6):
                    split = line.split()
                    muscles[(split[0],split[1])] = (float(split[2]),float(split[3]),float(split[4]),float(split[5]))      
                    muscle_connects[split[0]].add(split[1])
                    muscle_connects[split[1]].add(split[0])
            
            if line.strip() == "NODES":   
                mode = "NODES"    
            if line.strip() == "BONES":   
                mode = "BONES"    
                for i in nodes.keys():
                    bone_connects[i] = set() 
            if line.strip() == "MUSCLES": 
                mode = "MUSCLES"
                for i in nodes.keys():
                    muscle_connects[i] = set()

    return nodes, bones, muscles, bone_connects, muscle_connects, coords




def initialize_coords(bones, muscles, bone_connects, muscle_connects):
    
    coords = collections.OrderedDict()
    for node in bone_connects.keys():
        coords[node] = [0.0,0.0]
    for i in range(len(bone_connects)):
        nCon = len(list(bone_connects.values())[i]) + len(list(muscle_connects.values())[i])
        node = list(bone_connects.keys())[i]
        connects = list(bone_connects.values())[i]
        counter = 0
        for con in connects:
            if coords[con] == [0.0,0.0] and con != list(bone_connects.keys())[0]:
                angle = 2*np.pi/nCon * counter
                counter += 1
                coords[con][0] = np.cos(angle) * bones[(node,con)][1] + coords[node][0]
                coords[con][1] = np.sin(angle) * bones[(node,con)][1] + coords[node][1]
    for i in range(len(muscle_connects)):
        nCon = len(list(muscle_connects.values())[i]) + len(list(muscle_connects.values())[i])
        node = list(muscle_connects.keys())[i]
        connects = list(muscle_connects.values())[i]
        counter = 0
        for con in connects:
            if coords[con] == [0.0,0.0] and con != list(muscle_connects.keys())[0]:
                angle = 2*np.pi/nCon * counter
                counter += 1
                coords[con][0] = np.cos(angle) * muscles[(node,con)][1] + coords[node][0]
                coords[con][1] = np.sin(angle) * muscles[(node,con)][1] + coords[node][1]
    return coords
 
    
def center(coords):
    
    x_mean = 0
    y_mean = 0
    for i in coords.values():
        x_mean += i[0]
        y_mean += i[1]
    x_mean /= len(coords)
    y_mean /= len(coords)
    
    for i in coords.values():
        i[0] += 50 - x_mean # center to 50 percent
        i[1] += 50 - y_mean
        
    return coords
    
def energy(device,coords,bones,bones_indices,muscles,flexes,muscles_indices):
    
    energy = torch.zeros(1,dtype=torch.float32).to(device)
    
    i=0
    dists = torch.zeros(len(bones),dtype=torch.float32).to(device)
    for b in bones_indices:
        tmp = coords[b[1]]-coords[b[0]]
        dists[i] = torch.sqrt(torch.dot(tmp,tmp))
        i+=1
    energy += torch.sum((dists-bones[:,1])**2*bones[:,0])
    
    i=0
    dists = torch.zeros(len(muscles)).to(device)
    for m in muscles_indices:
        tmp = coords[m[1]]-coords[m[0]]
        dists[i] = torch.sqrt(torch.dot(tmp,tmp))
        i+=1
    energy += torch.sum((dists-muscles[:,1])**2/2*muscles[:,0]*flexes[:,0]+(dists-muscles[:,3])**2*muscles[:,2]*flexes[:,1])
    
    return energy
    
def minimize_energy(device,coords,bones,bones_indices,muscles,flexes,muscles_indices):
    
    optimizer = torch.optim.Adam(params=[coords], lr=world.LEARNING_RATE_MINIM, betas=(0.5, 0.999))
    
    for i in range(world.EPOCHS_MINIM):
        optimizer.zero_grad()
        ener = energy(device,coords,bones,bones_indices,muscles,flexes,muscles_indices)
        if(i%(world.EPOCHS_MINIM/10)==0): print("Minimizer Epoch: "+str(i)+" Energy: "+ str(ener))
        ener.backward()
        optimizer.step()
        
def apply_accelerations(device,coords,bones,bones_indices,muscles,flexes,muscles_indices,masses,velocities,timestep):
    vels = velocities
    
    i=0
    for b in bones_indices:
        tmp = coords[b[1]]-coords[b[0]]
        dist = torch.sqrt(torch.dot(tmp,tmp))
        uvec = tmp / dist
        force = (dist-bones[i,1])*bones[i,0]
        vels[b[0]] += force * uvec / masses[b[0]] * timestep
        vels[b[1]] -= force * uvec / masses[b[1]] * timestep   
        i+=1
        
    i=0
    for b in muscles_indices:
        tmp = coords[b[1]]-coords[b[0]]
        dist = torch.sqrt(torch.dot(tmp,tmp))
        uvec = tmp / dist
        force = (dist-muscles[i,1])*muscles[i,0]*flexes[i,0] + (dist-muscles[i,3])*muscles[i,2]*flexes[i,1]
        vels[b[0]] += force * uvec  / masses[b[0]] * timestep
        vels[b[1]] -= force * uvec  / masses[b[1]] * timestep   
        i+=1
    
    vels += torch.tensor([[0,-world.GRAVITY_CONSTANT * timestep] for i in vels],dtype=torch.float32).to(device)
    
    for i in range(len(coords)):
        if coords[i,1] < 0:
            vels[i,1] += world.GROUND_FORCE * timestep / masses[i]
    
    for i in range(len(vels)):
        vels[i] -= vels[i] * world.VISCOSITY * timestep / masses[i]
    
    return vels
    
def apply_accelerations_new(device,coords,bones,bones_indices,muscles,flexes,muscles_indices,masses,velocities,timestep):
    vels = velocities
    
    i=0
    for b in bones_indices:
        tmp = coords[b[1]]-coords[b[0]]
        dist = torch.sqrt(torch.dot(tmp,tmp))
        uvec = tmp / dist
        force = (dist-bones[i,1])*bones[i,0]
        vels[b[0]] += force * uvec / masses[b[0]] * timestep
        vels[b[1]] -= force * uvec / masses[b[1]] * timestep   
        i+=1
        
    i=0
    for b in muscles_indices:
        tmp = coords[b[1]]-coords[b[0]]
        dist = torch.sqrt(torch.dot(tmp,tmp))
        uvec = tmp / dist
        force = (dist-muscles[i,1])*muscles[i,0]*flexes[i,0] + (dist-muscles[i,3])*muscles[i,2]*flexes[i,1]
        vels[b[0]] += force * uvec  / masses[b[0]] * timestep
        vels[b[1]] -= force * uvec  / masses[b[1]] * timestep   
        i+=1
    
    vels += torch.tensor([[0,-world.GRAVITY_CONSTANT * timestep] for i in vels],dtype=torch.float32).to(device)
    
    for i in range(len(coords)):
        if coords[i,1] < 0:
            vels[i,1] += world.GROUND_FORCE * timestep / masses[i]
    
    for i in range(len(vels)):
        vels[i] -= vels[i] * world.VISCOSITY * timestep / masses[i]
    
    return vels
    
def integrate_leapfrog(device,coords,bones,bones_indices,muscles,flexes,muscles_indices,masses,velocities,timestep):
    
    coords += velocities * timestep
    
    return coords, apply_accelerations_new(device,coords,bones,bones_indices,muscles,flexes,muscles_indices,masses,velocities,timestep)


def draw(coords, bones, muscles):
    pygame.init()
    white = (255,255,255)
    black = (0,0,0)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)

    gameDisplay = pygame.display.set_mode((world.CANVAS_X,world.CANVAS_Y))
    gameDisplay.fill(black)
    
    coordinates = [[int(i[0]/100.0*world.CANVAS_X+0.5), world.CANVAS_Y - int(i[1]/100.0*world.CANVAS_Y+0.5)] for i in coords.values()]
    
    #print(coordinates)
    
    for i in coordinates:
        pygame.draw.circle(gameDisplay, green, i, 20)

    for i in bones.keys():
        pygame.draw.line(gameDisplay, white, coordinates[list(coords.keys()).index(i[0])], coordinates[list(coords.keys()).index(i[1])],5)
        
    for i in muscles.keys():
        pygame.draw.line(gameDisplay, red, coordinates[list(coords.keys()).index(i[0])], coordinates[list(coords.keys()).index(i[1])],5)

    def loop():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            pygame.display.update()
    loop()    
    
def watch_simulation(device,coords,bones,bones_indices,muscles,flexes,muscles_indices,masses,velocities,timestep,nSteps,delay=0):
    pygame.init()
    white = (255,255,255)
    black = (0,0,0)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)

    gameDisplay = pygame.display.set_mode((world.CANVAS_X,world.CANVAS_Y))
    
    
    coordinates = [[int(i[0]/100.0*world.CANVAS_X+0.5), world.CANVAS_Y - int(i[1]/100.0*world.CANVAS_Y+0.5)] for i in coords]
    
    #print(coordinates)
    def loop(start_flag):
        print_crds = coordinates
        crds = coords
        vels = velocities
        step = 0
        while True:
            if(step % world.SKIP_FRAMES == 0):
                gameDisplay.fill(black) 
                for i in print_crds:
                    pygame.draw.circle(gameDisplay, green, i, 20)
                for i in bones_indices:
                    pygame.draw.line(gameDisplay, white, print_crds[i[0]], print_crds[i[1]],5)
                for i in muscles_indices:
                    pygame.draw.line(gameDisplay, red, print_crds[i[0]], print_crds[i[1]],5)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            pygame.display.update()
            if start_flag and delay > 0:
                start_flag = False 
                time.sleep(delay)
            if step < nSteps:
                crds, vels = integrate_leapfrog(device,coords,bones,bones_indices,muscles,flexes,muscles_indices,masses,vels,timestep)
                print_crds = [[int(i[0]/100.0*world.CANVAS_X+0.5), world.CANVAS_Y - int(i[1]/100.0*world.CANVAS_Y+0.5)] for i in crds]
                policy(coords,flexes,timestep)
            step += 1
    loop(True)
    
    
def policy(coords,flexes,timestep):
    for flex in flexes:
        if np.random.uniform() < world.RANDOM_TENSION_CHANGE*timestep:
            if np.random.uniform() < world.RANDOM_CHANGE_TO_TENSION:
                flex[0]=0
                flex[1]=1
            else:
                flex[0]=1
                flex[1]=0
                
    

    
        
    
