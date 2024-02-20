from stable_baselines3.common.callbacks import BaseCallback

class CustomStop(BaseCallback):
    
    def __init__(self, verbose: int = 0):
        super(CustomStop, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        print(self.values)
        return True