import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class CustomCartPole(CartPoleEnv):
    def __init__(self, gravity=9.8, cart_mass=1.0, pole_mass=0.1,
                 pole_length=0.5, force_mag=10.0, **kwargs):

        super().__init__(**kwargs)

        # Override parameters
        self.gravity = gravity
        self.masscart = cart_mass
        self.masspole = pole_mass
        self.length = pole_length     # actually half the pole length
        self.force_mag = force_mag

        # Recompute dependent quantities
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length

from gymnasium.envs.registration import register

register(
    id="CustomCartPole-v0",
    entry_point="custom_cartpole:CustomCartPole",
)
