from collections import Counter, defaultdict
import csv
from itertools import accumulate
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
import random
import sys
import time

DEBUG_LEVEL = 0

INF = 10**9
NUM_WORKERS = 8

CARDS_CSV = "Pokemon TCG Pocket - Genetic Apex.csv"
POKE_EN2JA_JSON = "poke-en2ja.json"
PACK_NAMES = ["Mission", "Charizard", "Mewtwo", "Pikachu", "All"]
RARITY = ["", "♢", "♢♢", "♢♢♢", "♢♢♢♢", "☆", "☆☆", "☆☆☆", "♛"]
PRICE = [0, 35, 70, 150, 500, 400, 1250, 1500, 2500]
POINTS_PACK = 5
POINTS_MAX = 2500
PROB_GOD = 0.0005

poke_en2ja = {}
name = [""]
pack = [-1]
rarity = [-1]
prob1to3 = [0]
prob4 = [0]
prob5 = [0]
total_exp = [0]
cardprice = [0]
kantoid = [0]

kanto = []
bestprice_kanto = []
pack_kanto = []

num_dia = 0

pop1to3 = [[] for _ in range(len(PACK_NAMES) - 1)]
pop4 = [[] for _ in range(len(PACK_NAMES) - 1)]
pop5 = [[] for _ in range(len(PACK_NAMES) - 1)]

weight1to3 = [[] for _ in range(len(PACK_NAMES) - 1)]
weight4 = [[] for _ in range(len(PACK_NAMES) - 1)]
weight5 = [[] for _ in range(len(PACK_NAMES) - 1)]

cum_weight1to3 = [[] for _ in range(len(PACK_NAMES) - 1)]
cum_weight4 = [[] for _ in range(len(PACK_NAMES) - 1)]
cum_weight5 = [[] for _ in range(len(PACK_NAMES) - 1)]

god_pop = [[] for _ in range(len(PACK_NAMES) - 1)]

exp_kanto = [[] for _ in range(len(PACK_NAMES) - 1)]

init_exp_kanto = [-INF, 0, 0, 0]
init_exp_dia = [-INF, 0, 0, 0]
init_exp_all = [-INF, 0, 0, 0]


def percent2int(s):
    return round(float(s[:-1]) * 1000) if s else 0


def read_csv():
    global kanto, bestprice_kanto, exp_kanto, pack_kanto, num_dia, cum_weight1to3, cum_weight4, cum_weight5
    try:
        with open(CARDS_CSV, "r") as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
            kanto_str = rows[47][12].replace("Mr. Mime", "Mr.Mime").strip()
            kanto = [""] + list(
                map(
                    lambda p: poke_en2ja["Mr. Mime" if p == "Mr.Mime" else p],
                    kanto_str.split(),
                )
            )
            bestprice_kanto = [INF] * len(kanto)
            exp_kanto = [0] * len(kanto)
            pack_kanto = [4] * len(kanto)

            for i in range(1, len(rows)):
                row = rows[i]
                name_en = row[2]
                isex = name_en.endswith(" ex")
                poke = poke_en2ja[name_en[:-3]] if isex else poke_en2ja[name_en]
                name_ja = poke + ("ex" if isex else "")
                name.append(name_ja)
                pack.append(PACK_NAMES.index(row[3]))
                rarity.append(RARITY.index(row[4]))
                cardprice.append(PRICE[rarity[-1]])
                if 1 <= rarity[-1] <= 4:
                    num_dia += 1
                prob1to3.append(percent2int(row[5]))
                prob4.append(percent2int(row[6]))
                prob5.append(percent2int(row[7]))
                total_exp.append(prob1to3[-1] * 3 + prob4[-1] + prob5[-1])

                for p in [pack[-1]] if 1 <= pack[-1] <= 3 else [1, 2, 3]:
                    if prob1to3[-1]:
                        pop1to3[p].append(i)
                        weight1to3[p].append(prob1to3[-1])
                    if prob4[-1]:
                        pop4[p].append(i)
                        weight4[p].append(prob4[-1])
                    if prob5[-1]:
                        pop5[p].append(i)
                        weight5[p].append(prob5[-1])

                if 5 <= rarity[-1] <= 7:
                    god_pop[pack[-1]].append(i)
                elif rarity[-1] == 8:
                    god_pop[PACK_NAMES.index(name_en[:-3])].append(i)

                kid = kanto.index(poke) if poke in kanto else 0
                kantoid.append(kid)

                if kid > 0:
                    bestprice_kanto[kid] = min(bestprice_kanto[kid], cardprice[-1])
                if 1 <= pack[-1] <= 3:
                    exp = total_exp[-1]
                    if kid > 0 and name_ja != "イーブイ":
                        exp_kanto[kid] += exp
                        init_exp_kanto[pack[-1]] += exp
                        pack_kanto[kid] = pack[-1]
                    if 1 <= rarity[-1] <= 4:
                        init_exp_dia[pack[-1]] += exp
                    init_exp_all[pack[-1]] += exp

            for i in range(1, 4):
                cum_weight1to3[i] = list(accumulate(weight1to3[i]))
                cum_weight4[i] = list(accumulate(weight4[i]))
                cum_weight5[i] = list(accumulate(weight5[i]))

    except FileNotFoundError:
        print(
            'Download "Pokemon TCG Pocket - Genetic Apex.csv" from https://www.reddit.com/r/PTCGP/comments/1fwgwzi/comment/lw6hhbp/'
        )
        exit(1)


def draw(pack_id):
    if random.random() < PROB_GOD:
        return random.choices(god_pop[pack_id], k=5)
    else:
        result1to3 = random.choices(
            pop1to3[pack_id], cum_weights=cum_weight1to3[pack_id], k=3
        )
        result4 = random.choices(pop4[pack_id], cum_weights=cum_weight4[pack_id], k=1)
        result5 = random.choices(pop5[pack_id], cum_weights=cum_weight5[pack_id], k=1)
        return result1to3 + result4 + result5


def simulate(task_id, mode):
    assert mode in ["kanto", "dia1", "dia2", "all1", "all2"]
    num_req = 2 if mode[-1] == "2" else 1
    own = [0] * (
        len(kanto)
        if mode == "kanto"
        else num_dia + 1 if mode[:3] == "dia" else len(name)
    )
    remaining = {i: num_req for i in range(1, len(own))}
    remaining.pop(283, None)  # ミュウ
    points = 0
    points_needed = num_req * (
        sum(bestprice_kanto[1:])
        if mode == "kanto"
        else sum(cardprice[1 : num_dia + 1]) if mode[:3] == "dia" else sum(cardprice)
    )
    exp = (
        init_exp_kanto[:]
        if mode == "kanto"
        else init_exp_dia[:] if mode[:3] == "dia" else init_exp_all[:]
    )

    for turn in range(INF):
        if DEBUG_LEVEL >= 1:
            points_needed_naive = 0
            exp_naive = [-INF, 0, 0, 0]
            for x in remaining:
                if mode == "kanto":
                    points_needed_naive += bestprice_kanto[x] * remaining[x]
                    if pack_kanto[x] <= 3:
                        exp_naive[pack_kanto[x]] += exp_kanto[x]
                else:
                    points_needed_naive += cardprice[x] * remaining[x]
                    if pack[x] <= 3:
                        exp_naive[pack[x]] += total_exp[x]
            assert points_needed_naive == points_needed and exp_naive == exp
        if points >= points_needed:
            if DEBUG_LEVEL >= 1:
                print(
                    f"Buying all {sum(remaining.values())} remaining card(s) for {points_needed} pts"
                )
                for x in remaining:
                    print(
                        f"{kanto[x] if mode == "kanto" else name[x]} {bestprice_kanto[x] if mode == "kanto" else cardprice[x]} x {num_req - own[x]}"
                    )
            return turn
        pack_chosen = exp.index(max(exp))
        if DEBUG_LEVEL >= 2:
            print(f"[Pack {turn + 1}]")
            print(
                f"Own: {sum(min(num_req, x) for x in own[1:])} / {num_req * (len(own) - (2 if mode[:3] == "all" else 1))}"
            )
            print(f"Pts: {points} / {points_needed}")
            print(
                f"{PACK_NAMES[1]}: {exp[1]}, {PACK_NAMES[2]}: {exp[2]}, {PACK_NAMES[3]}: {exp[3]}"
            )
            print(f"Opening {PACK_NAMES[pack_chosen]}")
        results = draw(pack_chosen)
        for card in results:
            if DEBUG_LEVEL >= 2:
                print(
                    "{} A1-{:03} {} {}".format(
                        (
                            (
                                "NEW"
                                if kantoid[card] > 0 and own[kantoid[card]] == 0
                                else "   "
                            )
                            if mode == "kanto"
                            else (
                                f"{own[card]}>{own[card]+1}"
                                if card
                                <= (num_dia if mode[:3] == "dia" else len(name) - 1)
                                and own[card] < num_req
                                else "   "
                            )
                        ),
                        card,
                        name[card],
                        RARITY[rarity[card]],
                    )
                )
            if mode == "kanto":
                kid = kantoid[card]
                if kid > 0 and own[kid] == 0:
                    points_needed -= bestprice_kanto[kid]
                    if pack_kanto[kid] <= 3:
                        exp[pack_kanto[kid]] -= exp_kanto[kid]
                    remaining.pop(kid)
                own[kid] += 1
            else:
                if card <= (num_dia if mode[:3] == "dia" else len(name) - 1):
                    if own[card] < num_req:
                        points_needed -= cardprice[card]
                        if pack[card] <= 3 and own[card] == num_req - 1:
                            exp[pack_chosen] -= total_exp[card]
                        remaining[card] -= 1
                        if remaining[card] == 0:
                            remaining.pop(card)
                    own[card] += 1
        points = min(POINTS_MAX, points + POINTS_PACK)
        if points == POINTS_MAX and remaining:
            assert mode != "kanto"  # extremely unlikely
            rarity_priority = (
                [0, 1, 2, 4, 3] if mode[:3] == "dia" else [0, 1, 2, 4, 3, 6, 8, 5, 7]
            )
            chosen = max(
                remaining,
                key=lambda x: (
                    rarity_priority[rarity[x]],
                    remaining[x],
                    -exp[pack[x]] if pack[x] <= 3 else -INF,
                ),
            )
            price = cardprice[chosen]
            if DEBUG_LEVEL >= 1:
                print(f"After Pack {turn + 1}: {points} pts / {points_needed}")
                print(
                    f"{PACK_NAMES[1]}: {exp[1]}, {PACK_NAMES[2]}: {exp[2]}, {PACK_NAMES[3]}: {exp[3]}"
                )
                for card in remaining:
                    print(
                        card,
                        name[card],
                        RARITY[rarity[card]],
                        PACK_NAMES[pack[card]],
                        remaining[card],
                    )
                print("BUY", chosen, -price)
            points -= price
            points_needed -= price
            if pack[chosen] <= 3 and own[chosen] == num_req - 1:
                exp[pack[chosen]] -= total_exp[chosen]
            own[chosen] += 1
            remaining[chosen] -= 1
            if remaining[chosen] == 0:
                remaining.pop(chosen)


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d%H%M%S")
    try:
        with open(POKE_EN2JA_JSON, "r") as f:
            poke_en2ja = json.load(f)
    except FileNotFoundError:
        print(
            'Download "poke-en2ja.json" from https://github.com/evima0/pokepoke-comp-sim/blob/main/poke-en2ja.json'
        )
        exit(1)
    read_csv()
    assert len(kanto) == 150 + 1 and num_dia == 226 and len(name) == 286 + 1

    mode = sys.argv[1] if len(sys.argv) > 1 else "kanto"
    if mode not in ["kanto", "dia1", "dia2", "all1", "all2"]:
        print("Usage: python sim.py [kanto|dia1|dia2|all1|all2] [num_trials]")
        exit(1)
    num_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    DEBUG_LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    with Pool(NUM_WORKERS) as pool:
        start_time = time.time()
        results = pool.starmap(simulate, [(i, mode) for i in range(num_trials)])
        end_time = time.time()

    with open("{}-{}-{}.txt".format(mode, num_trials, timestamp), "w") as f:
        for t, c in sorted(Counter(results).items()):
            f.write(f"{t}: {c}\n")
        average = sum(results) / num_trials
        f.write("Average: {:.1f}\n".format(average))
        f.write(f"Min: {min(results)}\n")
        f.write(f"Median: {sorted(results)[num_trials // 2]}\n")
        f.write(f"95%: {sorted(results)[num_trials * 19 // 20]}\n")
        f.write(f"Max: {max(results)}\n")
        f.write(
            "SD: {:.1f}\n".format(
                (sum((x - average) ** 2 for x in results) / num_trials) ** 0.5
            )
        )
        f.write("Time: {:.3f} seconds\n".format(end_time - start_time))
    plt.figure(figsize=(10, 6))
    plt.hist(
        results,
        bins=[i for i in range(min(results), max(results) + 2)],
        edgecolor="black",
        alpha=0.7,
    )
    plt.grid(axis="y", alpha=0.75)
    plt.savefig("{}-{}-{}.png".format(mode, num_trials, timestamp))
