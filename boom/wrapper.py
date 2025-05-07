# -*- coding: utf-8 -*-
import copy

## TODO senzory: Funkce ktere koukaji na zbytek hry
# distance to mines
# distance to flag
# distance from middle of screen? (optimalization)
# fitness (minimize duration, max win)

import pygame
import numpy as np
import random
import math

from deap import base
from deap import creator
from deap import tools

pygame.font.init()

# -----------------------------------------------------------------------------
# Parametry hry
# -----------------------------------------------------------------------------

WIDTH, HEIGHT = 900, 500

WIN = pygame.display.set_mode((WIDTH, HEIGHT))

WHITE = (255, 255, 255)

TITLE = "Boom Master"
pygame.display.set_caption(TITLE)

FPS = 400
ME_VELOCITY = 5
MAX_MINE_VELOCITY = 3

BOOM_FONT = pygame.font.SysFont("comicsans", 100)
LEVEL_FONT = pygame.font.SysFont("comicsans", 20)

ENEMY_IMAGE = pygame.image.load("mine.png")
ME_IMAGE = pygame.image.load("me.png")
SEA_IMAGE = pygame.image.load("sea.png")
FLAG_IMAGE = pygame.image.load("flag.png")

ENEMY_SIZE = 50
ME_SIZE = 50

ENEMY = pygame.transform.scale(ENEMY_IMAGE, (ENEMY_SIZE, ENEMY_SIZE))
ME = pygame.transform.scale(ME_IMAGE, (ME_SIZE, ME_SIZE))
SEA = pygame.transform.scale(SEA_IMAGE, (WIDTH, HEIGHT))
FLAG = pygame.transform.scale(FLAG_IMAGE, (ME_SIZE, ME_SIZE))


# ----------------------------------------------------------------------------
# třídy objektů
# ----------------------------------------------------------------------------

# trida reprezentujici minu
class Mine:
    def __init__(self):

        # random x direction
        if random.random() > 0.5:
            self.dirx = 1
        else:
            self.dirx = -1

        # random y direction
        if random.random() > 0.5:
            self.diry = 1
        else:
            self.diry = -1

        x = random.randint(200, WIDTH - ENEMY_SIZE)
        y = random.randint(200, HEIGHT - ENEMY_SIZE)
        self.rect = pygame.Rect(x, y, ENEMY_SIZE, ENEMY_SIZE)

        self.velocity = random.randint(1, MAX_MINE_VELOCITY)


# trida reprezentujici me, tedy meho agenta
class Me:
    def __init__(self):
        self.rect = pygame.Rect(10, random.randint(1, 300), ME_SIZE, ME_SIZE)
        self.alive = True
        self.won = False
        self.timealive = 0
        self.sequence = []
        self.fitness = 0
        self.dist = 0


# třída reprezentující cíl = praporek
class Flag:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH - ME_SIZE, HEIGHT - ME_SIZE - 10, ME_SIZE, ME_SIZE)


# třída reprezentující nejlepšího jedince - hall of fame
class Hof:
    def __init__(self):
        self.sequence = []


# -----------------------------------------------------------------------------
# nastavení herního plánu
# -----------------------------------------------------------------------------


# rozestavi miny v danem poctu num
def set_mines(num):
    l = []
    for i in range(num):
        m = Mine()
        l.append(m)

    return l


# inicializuje me v poctu num na start
def set_mes(num):
    l = []
    for i in range(num):
        m = Me()
        l.append(m)

    return l


# zresetuje vsechny mes zpatky na start
def reset_mes(mes, pop):
    for i in range(len(pop)):
        me = mes[i]
        me.rect.x = 10
        me.rect.y = 10
        me.alive = True
        me.dist = 0
        me.won = False
        me.timealive = 0
        me.sequence = pop[i]
        me.fitness = 0


# -----------------------------------------------------------------------------
# senzorické funkce
# -----------------------------------------------------------------------------


# TODO

def mine_directional_threat_sensor(me: Me, mines, steps=15):
    threat_score = 0.0

    for mine in mines:
        future_mine = copy.deepcopy(mine)

        for step in range(1, steps + 1):
            handle_mine_movement(future_mine)
            expanded_me = me.rect.inflate(ME_VELOCITY * step * 3, ME_VELOCITY * step * 3)

            if expanded_me.colliderect(future_mine):
                score = 1.0 - (step / steps)
                threat_score = max(threat_score, score * 10)
                break
    return threat_score

def mine_dodge_probabilities(me: Me, mines: list[Mine], steps=15) -> list[float]:
    directions = {
        0: (0, -ME_VELOCITY),  # up
        1: (0, ME_VELOCITY),   # down
        2: (-ME_VELOCITY, 0),  # left
        3: (ME_VELOCITY, 0)    # right
    }

    safety = [1.0, 1.0, 1.0, 1.0]

    for dir_index, (dx, dy) in directions.items():
        # Simulate Me's future path if it moves in this direction
        future_me = me.rect.copy()

        for step in range(1, steps + 1):
            future_me.x += dx
            future_me.y += dy

            # Expand for worst-case movement from mines
            expanded_me = future_me.inflate(ME_VELOCITY * step * 2, ME_VELOCITY * step * 2)

            for mine in mines:
                future_mine = mine.rect.copy()
                for m_step in range(1, steps + 1):
                    future_mine.x += mine.dirx * mine.velocity
                    future_mine.y += mine.diry * mine.velocity

                    if expanded_me.colliderect(future_mine):
                        # Penalize this direction's safety
                        score = 1.0 - (m_step / steps)
                        safety[dir_index] = min(safety[dir_index], 1.0 - score)
                        break  # stop checking this mine for this step

    return safety  # values between 0 (certain death) and 1 (safe)

def nearest_mine_distance(me: Me, mines: list[Mine]):
    shortest_distance = math.inf
    for mine in mines:
        tmp = math.hypot(mine.rect.x - me.rect.x, mine.rect.y - me.rect.y)
        if  tmp < shortest_distance:
            shortest_distance = tmp
    return shortest_distance;

def flag_distance(me: Me, flag: Flag):
    return math.dist(me.rect.center, flag.rect.center)

def top_distance(me: Me):
    return me.rect.top / HEIGHT

def bottom_distance(me: Me):
    return  1 - (me.rect.bottom / HEIGHT)

def right_distance(me: Me):
    return me.rect.right / WIDTH

def left_distance(me: Me):
    return 1 - (me.rect.left / WIDTH)

# -----> ZDE je prostor pro vlastní senzorické funkce !!!!


# ---------------------------------------------------------------------------
# funkce řešící pohyb agentů
# ----------------------------------------------------------------------------


# konstoluje kolizi 1 agenta s minama, pokud je kolize vraci True
def me_collision(me, mines):
    for mine in mines:
        if me.rect.colliderect(mine.rect):
            # pygame.event.post(pygame.event.Event(ME_HIT))
            return True
    return False


# kolidujici agenti jsou zabiti, a jiz se nebudou vykreslovat
def mes_collision(mes, mines):
    for me in mes:
        if me.alive and not me.won:
            if me_collision(me, mines):
                me.alive = False


# vraci True, pokud jsou vsichni mrtvi Dave
def all_dead(mes):
    for me in mes:
        if me.alive:
            return False

    return True


# vrací True, pokud již nikdo nehraje - mes jsou mrtví nebo v cíli
def nobodys_playing(mes):
    for me in mes:
        if me.alive and not me.won:
            return False

    return True


# rika, zda agent dosel do cile
def me_won(me, flag):
    if me.rect.colliderect(flag.rect):
        return True

    return False


# vrací počet živých mes
def alive_mes_num(mes):
    c = 0
    for me in mes:
        if me.alive:
            c += 1
    return c


# vrací počet mes co vyhráli
def won_mes_num(mes):
    c = 0
    for me in mes:
        if me.won:
            c += 1
    return c


# resi pohyb miny
def handle_mine_movement(mine):
    if mine.dirx == -1 and mine.rect.x - mine.velocity < 0:
        mine.dirx = 1

    if mine.dirx == 1 and mine.rect.x + mine.rect.width + mine.velocity > WIDTH:
        mine.dirx = -1

    if mine.diry == -1 and mine.rect.y - mine.velocity < 0:
        mine.diry = 1

    if mine.diry == 1 and mine.rect.y + mine.rect.height + mine.velocity > HEIGHT:
        mine.diry = -1

    mine.rect.x += mine.dirx * mine.velocity
    mine.rect.y += mine.diry * mine.velocity


# resi pohyb min
def handle_mines_movement(mines):
    for mine in mines:
        handle_mine_movement(mine)


# ----------------------------------------------------------------------------
# vykreslovací funkce
# ----------------------------------------------------------------------------


# vykresleni okna
def draw_window(mes, mines, flag, level, generation, timer):
    WIN.blit(SEA, (0, 0))

    t = LEVEL_FONT.render("level: " + str(level), 1, WHITE)
    WIN.blit(t, (10, HEIGHT - 30))

    t = LEVEL_FONT.render("generation: " + str(generation), 1, WHITE)
    WIN.blit(t, (150, HEIGHT - 30))

    t = LEVEL_FONT.render("alive: " + str(alive_mes_num(mes)), 1, WHITE)
    WIN.blit(t, (350, HEIGHT - 30))

    t = LEVEL_FONT.render("won: " + str(won_mes_num(mes)), 1, WHITE)
    WIN.blit(t, (500, HEIGHT - 30))

    t = LEVEL_FONT.render("timer: " + str(timer), 1, WHITE)
    WIN.blit(t, (650, HEIGHT - 30))

    WIN.blit(FLAG, (flag.rect.x, flag.rect.y))

    # vykresleni min
    for mine in mines:
        WIN.blit(ENEMY, (mine.rect.x, mine.rect.y))

    # vykresleni me
    for me in mes:
        if me.alive:
            WIN.blit(ME, (me.rect.x, me.rect.y))

    pygame.display.update()


def draw_text(text):
    t = BOOM_FONT.render(text, 1, WHITE)
    WIN.blit(t, (WIDTH // 2, HEIGHT // 2))

    pygame.display.update()
    pygame.time.delay(1000)


# -----------------------------------------------------------------------------
# funkce reprezentující neuronovou síť, pro inp vstup a zadané váhy wei, vydá
# čtveřici výstupů pro nahoru, dolu, doleva, doprava
# ----------------------------------------------------------------------------


# <----- ZDE je místo vlastní funkci !!!!

def relu(x):
    return np.maximum(0, x)

def activ_sin(x):
    return np.sin(x % np.pi)

def activ_sinh(x):
    return np.sinh(x)

def activ_tanh(x):
    return np.tanh(x)

def temp_tanh(x):
    return np.tanh(x/400) * 350

def get_weights(wei):
    return np.array(wei[:4 * 4]).reshape((4, 4)), np.array(wei[4 * 4:]).reshape((4, 4))

# funkce reprezentující výpočet neuronové funkce
# funkce dostane na vstupu vstupy neuronové sítě inp, a váhy hran wei
# vrátí seznam hodnot výstupních neuronů
def nn_function(inp, wei):
    inputs = np.array(inp)  # shape: (5,)

    w1 = np.array(wei[:15]).reshape((5, 3))         # input -> hidden1
    w2 = np.array(wei[15:40]).reshape((5, 5))       # hidden1 -> hidden2
    w3 = np.array(wei[40:]).reshape((4, 5))         # hidden2 -> output

    if inputs[1] != 0:
        print(inputs)
    h1 = temp_tanh(w1 @ inputs)    # shape: (6,)
    h2 = activ_sin(w2 @ h1)        # shape: (4,)
    output = (w3 @ h2)               # shape: (4,)
    return int(np.argmax(output))

# naviguje jedince pomocí neuronové sítě a jeho vlastní sekvence v něm schované
def nn_navigate_me(me, inp):
    # TODO  <------ ZDE vlastní kód vyhodnocení výstupů z neuronové sítě !!!!!!
    ind = np.array(nn_function(inp, me.sequence))


    # nahoru, pokud není zeď
    if ind == 0 and me.rect.y - ME_VELOCITY > 0:
        me.rect.y -= ME_VELOCITY
        me.dist += ME_VELOCITY

    # dolu, pokud není zeď
    if ind == 1 and me.rect.y + me.rect.height + ME_VELOCITY < HEIGHT:
        me.rect.y += ME_VELOCITY
        me.dist += ME_VELOCITY

    # doleva, pokud není zeď
    if ind == 2 and me.rect.x - ME_VELOCITY > 0:
        me.rect.x -= ME_VELOCITY
        me.dist += ME_VELOCITY

    # doprava, pokud není zeď
    if ind == 3 and me.rect.x + me.rect.width + ME_VELOCITY < WIDTH:
        me.rect.x += ME_VELOCITY
        me.dist += ME_VELOCITY


# updatuje, zda me vyhrali
def check_mes_won(mes, flag):
    for me in mes:
        if me.alive and not me.won:
            if me_won(me, flag):
                me.won = True


# resi pohyb mes
def handle_mes_movement(mes, mines, flag):
    for me in mes:

        if me.alive and not me.won:
            # <----- ZDE  sbírání vstupů ze senzorů !!!
            # naplnit vstup in vstupy ze senzorů
            inp = []

            #inp.append(right_distance(me))
            #inp.append(top_distance(me))
            inp.append(flag_distance(me, flag))
            inp.append(mine_directional_threat_sensor(me, mines, 10))
           # inp.extend(mine_dodge_probabilities(me, mines, 10))
            inp.append(nearest_mine_distance(me, mines))

            nn_navigate_me(me, inp)


# updatuje timery jedinců
def update_mes_timers(mes, timer):
    for me in mes:
        if me.alive and not me.won:
            me.timealive = timer


# ---------------------------------------------------------------------------
# fitness funkce výpočty jednotlivců
# ----------------------------------------------------------------------------

def corner_penalty(me: Me, threshold=80):
    corners = [
        (0, 0),
        (WIDTH, 0),
        (0, HEIGHT),
    ]
    penalty = 0
    me_x, me_y = me.rect.center

    for cx, cy in corners:
        dist = math.hypot(cx - me_x, cy - me_y)
        if dist < threshold:
            penalty += (threshold - dist)

    return penalty * 5

# funkce pro výpočet fitness všech jedinců
def handle_mes_fitnesses(mes: list[Me]):
    total_scores = {
        "win_bonus": 0,
        "win_penalty": 0,
        "survival_reward": 0,
        "movement_reward": 0,
        "standing_still_penalty": 0,
        "diagonal_bonus": 0,
        "flag_proximity_reward": 0,
        "start_zone_penalty": 0,
        "corner_penalty": 0
    }

    for me in mes:
        score = 0
        me_x, me_y = me.rect.center

        # 1. Win bonus & time penalty
        if me.won:
            bonus = 50
            #penalty = me.timealive * 0.2
            score += bonus
            #score -= penalty
            total_scores["win_bonus"] += bonus
            #total_scores["win_penalty"] += penalty
        if me.alive:
            total_scores["survival_reward"] += 85
            score += 25

        # 2. Movement reward
        movement = me.dist * 0.01
        score += movement
        total_scores["movement_reward"] += movement

        # 3. Standing still penalty
        still_penalty = max(0, 400 - me.dist)
        score -= still_penalty
        total_scores["standing_still_penalty"] += still_penalty

        # 4. Diagonal movement bonus
        diagonal_alignment = 1 - abs(me_x - 1.8 * me_y) / WIDTH
        progress_reward = (me_x / WIDTH) + (me_y / HEIGHT)
        diagonal_bonus = diagonal_alignment * progress_reward * 20
        #score += diagonal_bonus
        total_scores["diagonal_bonus"] += diagonal_bonus

        # 5. Flag proximity reward
        goal_x = WIDTH - ME_SIZE
        goal_y = HEIGHT - ME_SIZE
        dist_to_flag = math.hypot(goal_x - me_x, (goal_y - me_y) * 2)
        max_dist = math.hypot(WIDTH, HEIGHT)
        flag_reward = (1 - dist_to_flag / max_dist) * 30
        score += flag_reward
        total_scores["flag_proximity_reward"] += flag_reward

        # 5. Start zone penalty
        dist_from_start = math.dist((5, 10), (me_x, me_y))
        start_penalty = max(0, HEIGHT + 50 - dist_from_start) * 0.5
        score -= start_penalty
        total_scores["start_zone_penalty"] += start_penalty

        # 6. Corner penalty
        total_scores["corner_penalty"] += corner_penalty(me)
        score -= corner_penalty(me, threshold=80)

        me.fitness = score

    n = len(mes)
    print("=== Fitness Category Averages ===")
    for key, total in total_scores.items():
        print(f"{key}: {total / n:.2f}")

    return [(me.fitness,) for me in mes]


# uloží do hof jedince s nejlepší fitness
def update_hof(hof, mes):
    l = [me.fitness for me in mes]
    ind = np.argmax(l)
    #w1 ,w2 = get_weights(mes[ind].sequence)
    # print(w1)
    # print(w2)
    hof.sequence = mes[ind].sequence.copy()

def temp_rand():
    return (random.random() * 2) - 1

# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------

def main():
    # =====================================================================
    # <----- ZDE Parametry nastavení evoluce !!!!!

    VELIKOST_POPULACE = 50
    EVO_STEPS = 5  # pocet kroku evoluce
    DELKA_JEDINCE = 60  # <--------- záleží na počtu vah a prahů u neuronů !!!!!
    NGEN = 800  # počet generací
    CXPB = 0.6  # pravděpodobnost crossoveru na páru
    MUTPB = 0.2  # pravděpodobnost mutace

    SIMSTEPS = 1000

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_rand", temp_rand)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rand, DELKA_JEDINCE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # vlastni random mutace
    def custom_mutate(individual, indpb, mutation_strength=0.5):
        for i in range(len(individual)):
            if random.random() < indpb:
                delta = random.uniform(-mutation_strength, mutation_strength)
                individual[i] += delta
        return (individual,)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("selectbest", tools.selBest)

    pop = toolbox.population(n=VELIKOST_POPULACE)

    # =====================================================================

    clock = pygame.time.Clock()

    # =====================================================================
    # testování hraním a z toho odvození fitness

    mines = []
    mes = set_mes(VELIKOST_POPULACE)
    flag = Flag()

    hof = Hof()

    run = True

    level = 5  # <--- ZDE nastavení obtížnosti počtu min !!!!!
    generation = 0

    evolving = True
    evolving2 = False
    timer = 0

    while run:

        clock.tick(FPS)

        # pokud evolvujeme pripravime na dalsi sadu testovani - zrestartujeme scenu
        if evolving:
            timer = 0
            generation += 1
            reset_mes(mes, pop)  # přiřadí sekvence z populace jedincům a dá je na start !!!!
            mines = set_mines(level)
            evolving = False

        timer += 1

        check_mes_won(mes, flag)
        handle_mes_movement(mes, mines, flag)

        handle_mines_movement(mines)

        mes_collision(mes, mines)

        if all_dead(mes):
            evolving = True
            # draw_text("Boom !!!")"""

        update_mes_timers(mes, timer)
        draw_window(mes, mines, flag, level, generation, timer)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # ---------------------------------------------------------------------
        # <---- ZDE druhá část evoluce po simulaci  !!!!!

        # druhá část evoluce po simulaci, když všichni dohrají, simulace končí 1000 krocích

        if timer >= SIMSTEPS or nobodys_playing(mes):

            # přepočítání fitness funkcí, dle dat uložených v jedinci
            handle_mes_fitnesses(mes)  # <--------- ZDE funkce výpočtu fitness !!!!

            update_hof(hof, mes)

            if generation >= NGEN:
                break

            if sum(list([me.won for me in mes])) >= VELIKOST_POPULACE / 2:
                level += 1

            # plot fitnes funkcí
            # ff = [me.fitness for me in mes]

            # print(ff)

            # přiřazení fitnessů z jedinců do populace
            # každý me si drží svou fitness, a každý me odpovídá jednomu jedinci v populaci
            for i in range(len(pop)):
                ind = pop[i]
                me = mes[i]
                ind.fitness.values = (me.fitness,)

            # selekce a genetické operace
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)

            pop[:] = offspring

            evolving = True

    # po vyskočení z cyklu aplikace vytiskne DNA sekvecni jedince s nejlepší fitness
    # a ukončí se
    print(hof.sequence)
    pygame.quit()


if __name__ == "__main__":
    main()