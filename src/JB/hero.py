
import csv
import os
import pandas as pd

class Hero:
    def __init__(self, name, player_type):
        self.name = name
        self.player_type = player_type

    def show_player(self):
        print("Player Name: " + self.name)
        print("Player Type: " + self.player_type)

# Fehler Cause: Forgetting to Instantiate an Object
#luke = Hero.show_player()

luke = Hero("Luke", "Mage") # instantiate an object --> luke
luke.show_player()

