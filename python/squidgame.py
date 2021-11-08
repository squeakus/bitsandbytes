from random import random


def main():
    current = 0
    finished = False
    rounds = 13
    players = 16

    for player in range(players):
        alive = True
        while alive:
            alive = jump()
            current += 1
            if current == rounds:
                print(f"Player {player} made it to the end!")
                finished = True
        if finished:
            break
        else:
            print(f"Player {player} fell off on {current}")


def jump():
    result = random()
    if result > 0.5:
        return True
    else:
        return False


if __name__ == "__main__":
    main()