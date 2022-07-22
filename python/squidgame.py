from random import random
import numpy as np
import matplotlib.pyplot as plt


def main():
    results = {"final_player": [], "successful_jumps": []}
    rounds = 13
    players = 16

    for i in range(10000):
        print(i)
        results = play_game(rounds, players, results)

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=results["final_player"])
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Player")
    plt.ylabel("Frequency")
    plt.title("How many players to complete the bridge")
    plt.savefig("final.png")

    plt.clf()
    n, bins, patches = plt.hist(x=results["successful_jumps"])
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Jumps")
    plt.ylabel("Frequency")
    plt.title("How many successful jumps on average")
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig("final2.png")


def play_game(rounds, players, results):
    prev = 0
    current = 0
    finished = False

    for player in range(players):
        alive = True
        while alive:
            alive = jump()
            current += 1
            if current == rounds:
                finished = True
        if finished:
            results["final_player"].append(player)
            break
        else:
            survived = current - prev
            results["successful_jumps"].append(survived)
            prev = current

    return results


def jump():
    result = random()
    if result > 0.5:
        return True
    else:
        return False


if __name__ == "__main__":
    main()