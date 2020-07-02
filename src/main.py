import factory
from train import Coach
import sys
if __name__ == "__main__":
    coach: Coach = factory.CoachFactory().init_coach_from_config("assets/train_config.json")
    coach.train()
