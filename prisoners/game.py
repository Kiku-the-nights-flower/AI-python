from random import randint, choice

from prisoners.evolver import play_game, tit_for_two_tats, titfortat, jesuswheel, \
    win_stay_lose_shift, always_coop, always_betray, kiku_the_nights_flwr_3, kiku_the_nights_flwr_2, \
    kiku_the_nights_flwr_1, kiku_the_nights_flwr

if __name__ == "__main__":
    won = 0
    opponents = []
    for i in range(100):
        opponent = jesuswheel
        result = play_game(kiku_the_nights_flwr, opponent, randint(100,150))
        print(result)
        if result[0] < result[1]:
            won += 1
    print(won)