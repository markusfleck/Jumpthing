GRAVITY_CONSTANT = 9.81 # The applied acceleretion due to gravity 
VISCOSITY = 0.4 # The viscosity of the medium. A low value mimics a gas, a high value a fluid
GROUND_FORCE = 1000.0 # The force which is applied when the animal hits the ground

CANVAS_X = 800 # the size of the screen display in pixels in x-direction
CANVAS_Y = 600 # the size of the screen display in pixels in y-direction

EPOCHS_MINIM = 2000 # how many energy mimimization steps to apply before the simulation
LEARNING_RATE_MINIM = 0.01 # the step size during the energy minimization

SKIP_FRAMES = 10 # how many integration steps to skip before drawing to the canvas (screen). A high value speeds up the simulation
