#==================================PATH==================================#

# Save directory
SAVE_DIR = './save/'

# Number 
NSPLITS = 5

#==================================Basic configuration of model==================================#
class CAEConfig:
    def __init__(self):
        self.epoch = 10
        self.batch_size = 50
        self.log_every = 1
        self.batch_norm = True # Use batch_norm or not 
        self.decay_every = 5 # Decay every decay_every
        self.decay_rate = 0.5 # every decay_every decay the learning rate