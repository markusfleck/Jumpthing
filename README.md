# Jumpthing
2D physics engine based on pyTorch for reinforcement learning

# 1) Introduction  
Based on techniques from Molecular Dynamics, this program simulates an animal made up of joints (nodes), bones and muscles. It offers custom building of animals (or machines if you wish) using a configuration text file. A reinforcement learning agent can be trained by customizing the policy(coords, flexes, timestep) function in tools.py. Customizing physics parameters is done in world.py. The code is extensively documented, so it should be instructional for a user to study. Demo videos are shipped with this repository.

# 2) Installation
This code needs pyTorch and pygame installed. It was tested with Python 3.7.9, pyTorch 1.3.1 and pygame 2.0.0.  Other versions might work just as well.

# 3) Building your own animal  
In bat.txt, a sample animal is shipped with this repository:  

    NODES
    M     1.0    0.0    10.0
    L1    1.0   -7.0   -3.0
    L2    1.0   -14.0   0.0
    L3    1.0   -21.0  -3.0
    R1    1.0    7.0   -3.0
    R2    1.0    14.0   0.0
    R3    1.0    21.0  -3.0
    
    BONES
    M  L1  1000.0 10.0
    L1 L2  1000.0 10.0
    L2 L3  1000.0 10.0
    M  R1  1000.0 10.0
    R1 R2  1000.0 10.0
    R2 R3  1000.0 10.0
    
    MUSCLES
    L1 R1 50.0 15.0 60.0 5.0
    L1 L3 50.0 15.0 60.0 5.0
    R1 R3 50.0 15.0 60.0 5.0


The "NODES" section specifies the joints. The first column specifies the name of the joint. The second column the  
mass of the joint. Note that only the nodes bear mass in this engine. The third column is the x-coordinate at the   
beginning of the simulation, the second is the y-coordinate. Units are percent of the screen size. The animal will be  
centered in the screen at the beginning of the simulation as well as energy minimized using the Adam optimizer of pyTorch. Therefore, the coordinates of the nodes should roughly match the length of the bones and muscles:

The "BONES" section, in graph theoretical jargon, defines edges(=bones) between the nodes(=joints). The first two columns define the joints which the bone connects. The third column defines a force constant for Hook's law as a resistance of the bone against elongation from its equilibrium (=natural) length. Obviously, for the bones to behave like a bones, this value should be high. The last column defines the equilibrium length in units of percent of the screen size.

The "MUSCLES" section is analogous to the "BONES" section, unless there is a flexed and a contracted state. In the flexed state, the force constant is column 3 and the equilibrium length is column 4. In the contracted state, the force constant is column 5 and the equilibrium length is column 6. A reinforcement agent controls the amount of contraction for the muscles. Note that the muscles are shorter and harder in the contracted state.

# 4) Modifying the physics
The constants for the physics are documented in and loaded from world.py. Feel free to change them. For example, if you feel a little bit like Q today, you might want to change the GRAVITY_CONSTANT by modifying its value in world.py.

# 5) Writing your own reinforcement learning agent
The policy(coords, flexes, timestep) function in tools.py is where you want to plug in your code. Right now,
this function only contains dummy code which performs random spasms.

# 6) Trivia
When Super Mario made his debut in 1981 in the Arcade Game Donkey Kong, his name was Jumpman.
 
