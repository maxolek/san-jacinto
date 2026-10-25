"""Paired-game (pentanomial) statistics in logistic and normalized Elo.

Games are played in opening pairs: the same opening twice with colors
reversed. The unit of evidence is the pair, not the game: the candidate's mean
score over a pair is one of LL, LD, DD / WL, WD, WW -> 0, .25, .5, .75, 1.
Every estimate here (Elo, confidence interval, LOS, normalized Elo, LLR) uses
the variance of these pair scores, which is smaller than the game-level
(trinomial) variance whenever an opening favors one color. Using the pair
variance is what makes the SPRT and the error bars honest.

The generalized likelihood maximizes the multinomial likelihood subject to each
hypothesis's expected score. Empty bins receive 1e-3 observations for numerical
stability, as in Fishtest:
https://github.com/official-stockfish/fishtest/blob/master/server/fishtest/stats/LLRcalc.py
Normalized Elo follows Van den Bergh, "Normalized Elo":
http://hardy.uhasselt.be/Fishtest/normalized_elo_practical.pdf
"""

import math
import re
from statistics import NormalDist

SUPPORT = (0.0, 0.25, 0.5, 0.75, 1.0)
LABELS = ("LL", "LD", "DD / WL", "WD", "WW")
DETAIL_LABELS = ("LL", "LD", "DD", "WL", "WD", "WW")
FINISHED_GAME = re.compile(r"Finished game (\d+) \((.+?) vs (.+?)\):\s*(\S+)")
NELO_PER_T = 800 / math.log(10)  # normalized Elo per unit of t-value, per game
Z95 = NormalDist().inv_cdf(0.975)


def elo_to_score(elo):
    return 1 / (1 + 10 ** (-elo / 400))

def score_to_elo(score):
    score = min(1 - 1e-9, max(1e-9, score))
    return 400 * math.log10(score / (1 - score))

def pair_bin(first_score, second_score):
    """Pentanomial bin (0..4) of a pair from the candidate's two game scores."""
    return round(2 * (first_score + second_score))

def pair_detail(first_score, second_score):
    """Like pair_bin, but separates DD (two draws) from WL (one win, one loss)."""
    total = first_score + second_score
    if total == 1:
        return "DD" if first_score == 0.5 else "WL"
    return {0: "LL", 0.5: "LD", 1.5: "WD", 2: "WW"}[total]

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
    """Pair-based Elo and normalized Elo with approximate 95% CIs, plus LOS.

    The CIs are fixed-sample intervals, not sequential guarantees: after an SPRT
    stops early they are slightly optimistic. Their sample size is the number
    of pairs, never the number of individual games.
    """
    empty = dict(score=None, elo_diff=None, elo_low=None, elo_high=None, los=None,
                 nelo=None, nelo_low=None, nelo_high=None, pair_stdev=None)
    if not sum(counts):
        return empty
    weights = _weights(counts)
    total = sum(weights)
    mean = sum(n * x for n, x in zip(weights, SUPPORT)) / total
    variance = sum(n * (x - mean) ** 2 for n, x in zip(weights, SUPPORT)) / total
    se = math.sqrt(variance / total)
    margin = Z95 * se
    # Per-game standard deviation implied by the pairs, and the t-value per game.
    game_sd = math.sqrt(2 * variance)
    nelo = (mean - 0.5) / game_sd * NELO_PER_T
    nelo_margin = Z95 * NELO_PER_T / math.sqrt(2 * total)
    return dict(score=mean, elo_diff=score_to_elo(mean),
                elo_low=score_to_elo(mean - margin), elo_high=score_to_elo(mean + margin),
                los=100 * NormalDist().cdf((mean - 0.5) / se),
                nelo=nelo, nelo_low=nelo - nelo_margin, nelo_high=nelo + nelo_margin,
                pair_stdev=math.sqrt(variance))

def trinomial_variance(wins, draws, losses):
    """Per-game score variance if games were independent (for comparison only)."""
    n = wins + draws + losses
    if not n:
        return None
    mean = (wins + 0.5 * draws) / n
    return (wins * (1 - mean) ** 2 + draws * (0.5 - mean) ** 2 + losses * mean ** 2) / n


class PairTally:
    """Pentanomial counts plus candidate W/D/L for any subset of pairs.

    Used for breakdowns (per core type, per opening, per level): each keeps its
    own counts so its Elo and error bars are computed from its own pairs.
    """

    def __init__(self):
        self.counts = [0] * 5
        self.detail = dict.fromkeys(DETAIL_LABELS, 0)
        self.wdl = [0, 0, 0]  # candidate wins, draws, losses

    def add_pair(self, first_score, second_score):
        self.counts[pair_bin(first_score, second_score)] += 1
        self.detail[pair_detail(first_score, second_score)] += 1
        for score in (first_score, second_score):
            self.wdl[{1: 0, 0.5: 1, 0: 2}[score]] += 1

    @property
    def pairs(self):
        return sum(self.counts)

    def summary(self):
        out = dict(pairs=self.pairs, pentanomial=self.counts.copy(), wins=self.wdl[0],
                   draws=self.wdl[1], losses=self.wdl[2], **estimate(self.counts))
        n = sum(self.wdl)
        out["draw_ratio"] = 100 * self.wdl[1] / n if n else None
        # Unregularized score for small breakdowns (estimate() pads empty bins).
        out["raw_score"] = (self.wdl[0] + 0.5 * self.wdl[1]) / n if n else None
        out["raw_elo"] = (score_to_elo(out["raw_score"])
                          if out["raw_score"] is not None and 0 < out["raw_score"] < 1 else None)
        return out

class PentanomialSPRT:
    """Consume opening pairs in a fixed order; freeze evidence at the first crossing.

    Game `n` belongs to pair (n - 1) // 2. The order of pairs must not depend on
    their results (use start order, never finish order): a pair may finish out
    of order, but it is not tested until all preceding pairs are complete or
    voided. Otherwise short decisive games would be over-represented when the
    test stops. Unknown/unfinished outcomes are never counted as draws.
    With sequential=False the bounds are never applied (result "fixed"): every
    pair is counted, for fixed-length matches that only estimate Elo.
    """

    def __init__(self, elo0, elo1, alpha=0.05, beta=0.05, sequential=True):
        if not all(math.isfinite(x) for x in (elo0, elo1, alpha, beta)):
            raise ValueError("SPRT parameters must be finite")
        if elo0 == elo1 or not all(0 < elo_to_score(x) < 1 for x in (elo0, elo1)):
            raise ValueError("Elo hypotheses must be distinct and have interior scores")
        if not (0 < alpha < 1 and 0 < beta < 1 and alpha + beta < 1):
            raise ValueError("alpha and beta must be positive with alpha + beta < 1")
        self.elo0, self.elo1 = elo0, elo1
        self.alpha, self.beta = alpha, beta
        self.sequential = sequential
        self.lbound = math.log(beta / (1 - alpha))
        self.ubound = math.log((1 - beta) / alpha)
        self.tally = PairTally()
        self.games = {}
        self.voided = set()
        self.next_pair = 0
        self.pairs_tested = 0
        self.pairs_voided = 0
        self.wdl = {color: [0, 0, 0] for color in ("white", "black")}
        self.llr = 0.0
        self.result = "inconclusive" if sequential else "fixed"

    @property 
    def counts(self):
        return self.tally.counts

    @property
    def decided(self):
        """An SPRT bound has been crossed (never for fixed-length matches)."""
        return self.sequential and self.result != "inconclusive"

    def _advance(self):
        updates = []
        while not self.decided:
            if self.next_pair in self.voided:
                self.next_pair += 1
                self.pairs_voided += 1
                continue
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
            self.tally.add_pair(*scores)
            self.next_pair += 1
            self.pairs_tested += 1
            self.llr = compute_llr(self.counts, self.elo0, self.elo1)
            if self.sequential and self.llr >= self.ubound:
                self.result = "pass"
            elif self.sequential and self.llr <= self.lbound:
                self.result = "fail"
            updates.append(self.snapshot())
        return updates

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
        return self._advance()

    def void_pair(self, pair_index):
        """Exclude a pair that can never complete (e.g. its match process died)
        
        the pair keeps its place in the order, so later pairs are not reordered
        """
        if pair_index >= self.next_pair and pair_index not in self.voided:
            self.voided.add(pair_index)
        return self._advance()

    def snapshot(self):
        tested = self.pairs_tested
        pending = sum(1 for n in self.games if (n - 1) // 2 >= self.next_pair)
        snapshot = dict(model="pentanomial-logistic", pentanomial=self.counts.copy(),
                        pentanomial_detail=dict(self.tally.detail),
                        pairs=tested, games_played=2 * tested,
                        completed_games=len(self.games), untested_games=pending,
                        voided_pairs=self.pairs_voided,
                        elo0=self.elo0, elo1=self.elo1, alpha=self.alpha, beta=self.beta,
                        llr=self.llr, lbound=self.lbound, ubound=self.ubound, result=self.result,
                        **estimate(self.counts))
        for i, name in enumerate(("wins", "draws", "losses")):
            snapshot[f"candidate_{name}"] = sum(values[i] for values in self.wdl.values())
            for color, values in self.wdl.items():
                snapshot[f"candidate_{color}_{name}"] = values[i]
        n = snapshot["games_played"]
        snapshot["draw_ratio"] = 100 * snapshot["candidate_draws"] / n if n else None
        tri = trinomial_variance(snapshot["candidate_wins"], snapshot["candidate_draws"],
                                 snapshot["candidate_losses"])
        pair_sd = snapshot["pair_stdev"]
        # < 1: color-reversed pairs cancel opening bias, so fewer games are needed
        snapshot["variance_ratio"] = (2 * pair_sd ** 2 / tri) if tri and pair_sd is not None else None
        # Distance to the nearer bound along the current average LLR drift per pair.
        snapshot["llr_progress"] = (self.llr / self.ubound if self.llr >= 0 else self.llr / self.lbound)
        snapshot["eta_pairs"] = None
        if self.result == "inconclusive" and tested >= 20 and self.llr:
            drift = self.llr / tested
            target = self.ubound if drift > 0 else self.lbound
            snapshot["eta_pairs"] = max(0, round((target - self.llr) / drift))
        return snapshot
