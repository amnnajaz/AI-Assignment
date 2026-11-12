"""
Part B: Isolation on a 5x5 board with an alpha-beta AI.

Rules implemented
    * 5x5 grid, AI (player 0) starts at (0,0), Human (player 1) at (4,4). AI moves first.
    * Players move like a chess QUEEN (any distance along row/column/diagonal).
      A move cannot cross or land on a destroyed cell or on the other player's token.
    * The cell a player just LEFT is destroyed permanently.
    * A player with no legal move on his/her turn loses.

Contents
    State                 - immutable game state (bitboards)
    alphabeta / minimax   - depth-limited search (default ceiling = 4 plies)
    h_mobility            - Heuristic 1: (#my moves - #opponent moves)
    h_territory           - Heuristic 2: Voronoi territory (cells I can reach first)
    play_vs_human         - interactive game in the terminal   (python isolation.py --play)
    run_experiments       - benchmark used for the report       (python isolation.py)
"""
import argparse
import random
import time

N = 5
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
WIN = 1000          # magnitude of a terminal (win/loss) score
MAX_DEPTH = 4       # depth ceiling required by the assignment


def bit(cell):
    """Bit mask of a (row, col) cell."""
    return 1 << (cell[0] * N + cell[1])


# Game state
class State:
    """Immutable state: positions of both tokens, destroyed-cell bitmask, whose turn.
    player 0 = AI, player 1 = Human."""
    __slots__ = ("pos", "destroyed", "turn")

    def __init__(self, pos=((0, 0), (N - 1, N - 1)), destroyed=0, turn=0):
        self.pos = pos
        self.destroyed = destroyed
        self.turn = turn

    def moves(self, player=None):
        """All legal destination cells for `player` (default: the player to move)."""
        player = self.turn if player is None else player
        r0, c0 = self.pos[player]
        blocked = self.destroyed | bit(self.pos[1 - player])   # opponent token blocks too
        out = []
        for dr, dc in DIRECTIONS:
            r, c = r0 + dr, c0 + dc
            while 0 <= r < N and 0 <= c < N and not (blocked & bit((r, c))):
                out.append((r, c))
                r += dr
                c += dc
        return out

    def apply(self, move):
        """Return the successor state: mover goes to `move`, the vacated cell is destroyed."""
        p = self.turn
        new_pos = list(self.pos)
        old = new_pos[p]
        new_pos[p] = move
        return State(tuple(new_pos), self.destroyed | bit(old), 1 - p)

    def is_over(self):
        return not self.moves()

    def loser(self):
        return self.turn if self.is_over() else None

    def render(self, labels=("A", "H")):
        rows = []
        for r in range(N):
            line = []
            for c in range(N):
                if self.pos[0] == (r, c):
                    line.append(labels[0])
                elif self.pos[1] == (r, c):
                    line.append(labels[1])
                elif self.destroyed & bit((r, c)):
                    line.append("X")
                else:
                    line.append(".")
            rows.append(f"{r} " + " ".join(line))
        return "  " + " ".join(str(c) for c in range(N)) + "\n" + "\n".join(rows)


# Heuristic evaluation functions (always scored from `me`'s point of view)
def h_mobility(state, me):
    """Heuristic 1 - linear mobility differential: #my options - #opponent options."""
    return len(state.moves(me)) - len(state.moves(1 - me))


def _bfs_distances(state, player):
    """Number of queen moves needed to reach every free cell, ignoring the other
    player's future moves (the other token is treated as a wall)."""
    blocked = state.destroyed | bit(state.pos[1 - player])
    dist = {state.pos[player]: 0}
    frontier = [state.pos[player]]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for (r0, c0) in frontier:
            for dr, dc in DIRECTIONS:
                r, c = r0 + dr, c0 + dc
                while 0 <= r < N and 0 <= c < N and not (blocked & bit((r, c))):
                    if (r, c) not in dist:
                        dist[(r, c)] = d
                        nxt.append((r, c))
                    r += dr
                    c += dc
        frontier = nxt
    return dist


def h_territory(state, me):
    """Heuristic 2 - Voronoi territory (custom metric).
    A free cell belongs to the player who can reach it in fewer queen moves.
    Ties go to the player whose turn it is (they get there first).
        score = (#cells I own) - (#cells opponent owns)
    Cells reachable by only one player count fully for that player, which
    rewards partitioning the board into a large private region - the key idea
    of the Isolation end-game."""
    d_me = _bfs_distances(state, me)
    d_op = _bfs_distances(state, 1 - me)
    mine = theirs = 0
    for cell in set(d_me) | set(d_op):
        if cell == state.pos[me] or cell == state.pos[1 - me]:
            continue
        a, b = d_me.get(cell, 99), d_op.get(cell, 99)
        if a < b or (a == b and state.turn == me):
            mine += 1
        elif b < a or (a == b and state.turn != me):
            theirs += 1
    return mine - theirs


HEURISTICS = {"H1-mobility": h_mobility, "H2-territory": h_territory}


# Search

class SearchStats:
    def __init__(self):
        self.nodes = 0          # nodes expanded (calls of the search function)
        self.cutoffs = 0        # number of alpha >= beta prunings


def alphabeta(state, depth, alpha, beta, me, heuristic, stats):
    """Depth-limited minimax with alpha-beta pruning.
    `me` is the player we are choosing a move for (MAX); the other is MIN.
    Terminal nodes score +-(WIN + depth) so that quicker wins / slower losses are preferred.
    Leaf nodes (depth == 0) are scored by `heuristic`.
    Returns (value, best_move)."""
    stats.nodes += 1
    legal = state.moves()
    if not legal:                                   # player to move has lost
        return (-(WIN + depth) if state.turn == me else (WIN + depth)), None
    if depth == 0:
        return heuristic(state, me), None

    best_move = None
    if state.turn == me:                            # MAX node
        value = float("-inf")
        for mv in legal:
            child, _ = alphabeta(state.apply(mv), depth - 1, alpha, beta, me, heuristic, stats)
            if child > value:
                value, best_move = child, mv
            alpha = max(alpha, value)
            if alpha >= beta:                       # cutoff condition
                stats.cutoffs += 1
                break
        return value, best_move
    value = float("inf")                            # MIN node
    for mv in legal:
        child, _ = alphabeta(state.apply(mv), depth - 1, alpha, beta, me, heuristic, stats)
        if child < value:
            value, best_move = child, mv
        beta = min(beta, value)
        if alpha >= beta:
            stats.cutoffs += 1
            break
    return value, best_move


def minimax(state, depth, me, heuristic, stats):
    """Plain depth-limited minimax (no pruning) - used only to measure how many
    nodes alpha-beta saves."""
    stats.nodes += 1
    legal = state.moves()
    if not legal:
        return (-(WIN + depth) if state.turn == me else (WIN + depth)), None
    if depth == 0:
        return heuristic(state, me), None
    best_move = None
    maximizing = state.turn == me
    value = float("-inf") if maximizing else float("inf")
    for mv in legal:
        child, _ = minimax(state.apply(mv), depth - 1, me, heuristic, stats)
        if (maximizing and child > value) or (not maximizing and child < value):
            value, best_move = child, mv
    return value, best_move


def choose_move(state, heuristic, depth=MAX_DEPTH, stats=None):
    """Pick the best move for the player to move using alpha-beta to `depth` plies."""
    stats = stats or SearchStats()
    _, mv = alphabeta(state, depth, float("-inf"), float("inf"), state.turn, heuristic, stats)
    return mv


# Playing games

def play_game(agents, random_opening_plies=0, rng=None, depth=MAX_DEPTH, verbose=False):
    """Play one full game between two agents. `agents[p]` is a heuristic function
    (alpha-beta agent) or the string 'random'. The first `random_opening_plies`
    plies are played randomly to create game variety.
    Returns dict(winner, plies, nodes[2], time[2])."""
    rng = rng or random.Random(0)
    s = State()
    nodes = [0, 0]
    times = [0.0, 0.0]
    plies = 0
    while not s.is_over():
        p = s.turn
        legal = s.moves()
        if plies < random_opening_plies or agents[p] == "random":
            mv = rng.choice(legal)
        else:
            st = SearchStats()
            t0 = time.perf_counter()
            mv = choose_move(s, agents[p], depth, st)
            times[p] += time.perf_counter() - t0
            nodes[p] += st.nodes
        s = s.apply(mv)
        plies += 1
        if verbose:
            print(f"ply {plies}: player {p} -> {mv}")
            print(s.render())
    return dict(winner=1 - s.loser(), plies=plies, nodes=nodes, time=times)


def play_vs_human(heuristic=h_territory, depth=MAX_DEPTH):
    """Interactive game: you are 'H' at (4,4); the AI is 'A' at (0,0) and moves first."""
    s = State()
    print("ISOLATION 5x5 - queen moves; the cell you leave is destroyed (X).")
    print("Enter moves as 'row col'. The player who cannot move loses.\n")
    while not s.is_over():
        print(s.render(), "\n")
        if s.turn == 0:
            mv = choose_move(s, heuristic, depth)
            print(f"AI moves to {mv}\n")
        else:
            legal = s.moves()
            print("Your legal moves:", legal)
            while True:
                try:
                    r, c = map(int, input("Your move (row col): ").split())
                    if (r, c) in legal:
                        mv = (r, c)
                        break
                except ValueError:
                    pass
                print("Illegal move, try again.")
        s = s.apply(mv)
    print(s.render())
    print("\nYou win!" if s.loser() == 0 else "\nAI wins!")


# Experiments for the report

def run_experiments(games_per_side=50, seed=2026, verbose=True):
    """Return all numbers used in the report."""
    res = {}
    rng = random.Random(seed)

    # (1) pruning efficiency at the initial position, depth 1..4 (both heuristics)
    prune = []
    for d in range(1, MAX_DEPTH + 1):
        s0 = State()
        a, m = SearchStats(), SearchStats()
        v1, mv1 = alphabeta(s0, d, float("-inf"), float("inf"), 0, h_mobility, a)
        v2, mv2 = minimax(s0, d, 0, h_mobility, m)
        assert v1 == v2                      # alpha-beta must return the minimax value
        prune.append((d, m.nodes, a.nodes, a.cutoffs))
    res["pruning"] = prune

    # (2) head-to-head tournament, random 2-ply openings (1 random move each) for variety
    def tournament(h_a, h_b, n):
        wins_a = 0
        plies, nodes_a, nodes_b, t_a, t_b = [], [], [], [], []
        for i in range(n):
            for a_is_first in (True, False):
                agents = (h_a, h_b) if a_is_first else (h_b, h_a)
                local = random.Random(rng.random())
                g = play_game(agents, random_opening_plies=2, rng=local)
                a_idx = 0 if a_is_first else 1
                wins_a += g["winner"] == a_idx
                plies.append(g["plies"])
                nodes_a.append(g["nodes"][a_idx]); nodes_b.append(g["nodes"][1 - a_idx])
                t_a.append(g["time"][a_idx])
                t_b.append(g["time"][1 - a_idx])
        total = 2 * n
        return dict(games=total, wins_a=wins_a, wins_b=total - wins_a,
                    avg_plies=sum(plies) / total,
                    avg_nodes_a=sum(nodes_a) / total, avg_nodes_b=sum(nodes_b) / total,
                    avg_time_a=sum(t_a) / total, avg_time_b=sum(t_b) / total)

    res["H2_vs_H1"] = tournament(h_territory, h_mobility, games_per_side)

    # (3) each heuristic vs a random mover (sanity check)
    for name, h in HEURISTICS.items():
        w = 0
        n = 2 * games_per_side
        for i in range(games_per_side):
            for first in (True, False):
                agents = (h, "random") if first else ("random", h)
                g = play_game(agents, rng=random.Random(rng.random()))
                w += g["winner"] == (0 if first else 1)
        res[f"{name}_vs_random"] = (w, n)

    # (4) one fully deterministic showcase game (no random opening)
    res["showcase_H1_vs_H2"] = play_game((h_mobility, h_territory))
    res["showcase_H2_vs_H1"] = play_game((h_territory, h_mobility))

    if verbose:
        print("=" * 72)
        print("PART B - Isolation 5x5 (queen moves), alpha-beta depth <= 4")
        print("=" * 72)
        print("\nPruning at the start position (heuristic H1):")
        print("  depth  minimax_nodes  alphabeta_nodes  cutoffs")
        for d, mn, an, co in res["pruning"]:
            print(f"  {d:5d}  {mn:13d}  {an:15d}  {co:7d}")
        print("\nH2-territory vs H1-mobility:", res["H2_vs_H1"])
        for name in HEURISTICS:
            print(f"{name} vs random: {res[name + '_vs_random']}")
        print("Showcase (H1 as P0, H2 as P1): winner =", res["showcase_H1_vs_H2"]["winner"])
        print("Showcase (H2 as P0, H1 as P1): winner =", res["showcase_H2_vs_H1"]["winner"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true", help="play against the AI in the terminal")
    ap.add_argument("--heuristic", choices=list(HEURISTICS), default="H2-territory")
    ap.add_argument("--games", type=int, default=50, help="games per colour in the tournament")
    args = ap.parse_args()
    if args.play:
        play_vs_human(HEURISTICS[args.heuristic])
    else:
        run_experiments(args.games)
