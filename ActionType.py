from enum import Enum


class ActionType(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    SKIP = 4
    CIRCUMNAVIGATE = 5

    @staticmethod
    def get_action_name(action_number):
        action_dict = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right',
            4: 'skip',
            5: 'circumnavigate'
        }
        return action_dict.get(action_number, "Unknown Action")