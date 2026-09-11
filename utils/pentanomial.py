"""Paired-game SPRT in logistic Elo, independent of cutechess and plotting.

Support is the candidate's mean score in a color-reversed opening pair:
LL, LD, DD/WL, WD, WW -> 0, .25, .5, .75, 1. The generalized likelihood
maximizes multinomial likelihood subject to each hypothesis's expected score.
Empty bins receive 1e-3 observations for numerical stability, as in Fishtest:
https://github.com/official-stockfish/fishtest/blob/master/server/fishtest/stats/LLRcalc.py
"""

import math
import re
from statistics import NormalDist

SUPPORT = (0.0, 0.25, 0.5, 0.75, 1.0)
LABELS = ("LL", "LD", "DD / WL", "WD", "WW")
FINISHED_GAME = re.compile(r"Finished game (\d+) \((.+?) vs (.+?)\):\s*(\S+)")


def elo_to_score(elo):
    return 1 / (1 + 10 ** (-elo / 400))


def score_to_elo(score):
    score = min(1 - 1e-9, max(1e-9, score))
    return 400 * math.log10(score / (1 - score))


def _weights(counts):
    if len(counts) != 5 or any(n < 0 or not math.isfinite(n) for n in counts):
        raise ValueError("Expected five nonnegative finite pentanomial counts")
    return [n if n else 1e-3 for n in counts]


def constrained_probabilities(weights, mean):
    """Solve the Lagrange multiplier equation on its positive-probability domain."""
    if not 0 < mean < 1:
        raise ValueError("Hypothesis score must be strictly between zero and one")
    total = sum(weights)
    frequencies = [n / total for n in weights]
    offsets = [x - mean for x in SUPPORT]
    lo, hi = -1 / (1 - mean), 1 / mean
    for _ in range(90):
        multiplier = (lo + hi) / 2
        value = sum(p * d / (1 + multiplier * d) for p, d in zip(frequencies, offsets))
        if value > 0:
            lo = multiplier
        else:
            hi = multiplier
        if hi - lo < 1e-13:
            break
    return [p / (1 + multiplier * d) for p, d in zip(frequencies, offsets)]


def compute_llr(counts, elo0, elo1):
    weights = _weights(counts)
    if not sum(counts):
        return 0.0
    p0 = constrained_probabilities(weights, elo_to_score(elo0))
    p1 = constrained_probabilities(weights, elo_to_score(elo1))
    return sum(n * math.log(b / a) for n, a, b in zip(weights, p0, p1))


def estimate(counts):
    """Pair-based Elo, approximate fixed-sample 95% CI and normal-approximation LOS.

    The CI is descriptive, not a sequential confidence guarantee. Its sample
    size is the number of pairs, never the number of individual games.
    """
    weights = _weights(counts)
    if not sum(counts):
        return dict(score=None, elo_diff=None, elo_low=None, elo_high=None, los=None)
    total = sum(weights)
    mean = sum(n * x for n, x in zip(weights, SUPPORT)) / total
    variance = sum(n * (x - mean) ** 2 for n, x in zip(weights, SUPPORT)) / total
    se = math.sqrt(variance / total)
    margin = NormalDist().inv_cdf(0.975) * se
    return dict(score=mean, elo_diff=score_to_elo(mean),
                elo_low=score_to_elo(mean - margin), elo_high=score_to_elo(mean + margin),
                los=100 * NormalDist().cdf((mean - 0.5) / se))


class PentanomialSPRT:
    """Consume consecutive opening pairs; freeze evidence at the first crossing.

    A pair may finish out of order, but it is not tested until all preceding
    pairs are complete. Unknown/unfinished outcomes are never counted as draws.
    """

    def __init__(self, elo0, elo1, alpha=0.05, beta=0.05):
        if not all(math.isfinite(x) for x in (elo0, elo1, alpha, beta)):
            raise ValueError("SPRT parameters must be finite")
        if elo0 == elo1 or not all(0 < elo_to_score(x) < 1 for x in (elo0, elo1)):
            raise ValueError("Elo hypotheses must be distinct and have interior scores")
        if not (0 < alpha < 1 and 0 < beta < 1 and alpha + beta < 1):
            raise ValueError("alpha and beta must be positive with alpha + beta < 1")
        self.elo0, self.elo1 = elo0, elo1
        self.lbound = math.log(beta / (1 - alpha))
        self.ubound = math.log((1 - beta) / alpha)
        self.counts = [0] * 5
        self.games = {}
        self.next_pair = 0
        self.wdl = {color: [0, 0, 0] for color in ("white", "black")}
        self.llr = 0.0
        self.result = "inconclusive"

    def add_game(self, number, white, black, result):
        if number < 1 or {white, black} != {"Candidate", "Baseline"}:
            raise ValueError("Unexpected game identity")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            return []
        game = (white == "Candidate", result)
        if number in self.games:
            if self.games[number] != game:
                raise ValueError(f"Conflicting result for game {number}")
            return []
        self.games[number] = game
        updates = []
        while self.result == "inconclusive":
            first = 2 * self.next_pair + 1
            if first not in self.games or first + 1 not in self.games:
                break
            pair = [self.games[first], self.games[first + 1]]
            if pair[0][0] == pair[1][0]:
                raise ValueError(f"Opening pair {self.next_pair + 1} did not reverse colors")
            scores = []
            for candidate_white, outcome in pair:
                score = 0.5 if outcome == "1/2-1/2" else float((outcome == "1-0") == candidate_white)
                scores.append(score)
                self.wdl["white" if candidate_white else "black"][{1: 0, 0.5: 1, 0: 2}[score]] += 1
            self.counts[round(2 * sum(scores))] += 1
            self.next_pair += 1
            self.llr = compute_llr(self.counts, self.elo0, self.elo1)
            if self.llr >= self.ubound:
                self.result = "pass"
            elif self.llr <= self.lbound:
                self.result = "fail"
            updates.append(self.snapshot())
        return updates

    def snapshot(self):
        snapshot = dict(model="pentanomial-logistic", pentanomial=self.counts.copy(),
                        pairs=self.next_pair, games_played=2 * self.next_pair,
                        completed_games=len(self.games), untested_games=len(self.games) - 2 * self.next_pair,
                        elo0=self.elo0, elo1=self.elo1, llr=self.llr,
                        lbound=self.lbound, ubound=self.ubound, result=self.result,
                        **estimate(self.counts))
        for i, name in enumerate(("wins", "draws", "losses")):
            snapshot[f"candidate_{name}"] = sum(values[i] for values in self.wdl.values())
            for color, values in self.wdl.items():
                snapshot[f"candidate_{color}_{name}"] = values[i]
        n = snapshot["games_played"]
        snapshot["draw_ratio"] = 100 * snapshot["candidate_draws"] / n if n else None
        return snapshot
