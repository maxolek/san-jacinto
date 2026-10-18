"""Maximum-likelihood Elo ratings from all stored game results.

One fit over every result replaces per-run K-factor updates: the ratings do not
depend on the order in which matches were played, are consistent across the
whole pool and come with error bars. This is the Bradley-Terry model as fitted
by Ordo and BayesElo (without BayesElo's separate draw parameter).

Model: logistic Elo. Per game, player i is expected to score
    p_ij = 1 / (1 + 10 ** (-(r_i - r_j) / 400))
against player j, a draw counting as half a win and half a loss. With
S_ij = W + D/2 points scored in N_ij games, the ratings maximize
    LL = sum over pairs of S_ij log(p_ij) + (N_ij - S_ij) log(1 - p_ij)
by Newton-Raphson with a backtracking line search, in natural log-odds units
theta = r ln(10) / 400, with each component's anchor held fixed.

Prior: prior_draws virtual draws are added to every pair of players that played
each other, as in BayesElo. They keep 100% and 0% scores finite and barely move
well-sampled pairs. With prior_draws=0 a group that scored 100% against the rest
of its component has no finite rating, and fit_ratings raises ValueError.

Components: players linked by games form components that are fitted separately,
each with one player (its anchor) held at anchor_elo. Ratings in different
components are not comparable.

Errors: the covariance is the inverse of the negative Hessian of LL at the
optimum (prior included), so standard errors are relative to the component's
anchor. They treat games as independent. Paired-opening games are slightly
negatively correlated for a given engine, so these errors are slightly
conservative; scoring a draw as half a win and half a loss also overstates the
per-game variance when draws are common, which makes them more conservative.
"""

import math
from dataclasses import dataclass, field

import numpy as np

ELO_PER_LOGIT = 400 / math.log(10)  # Elo per unit of natural log-odds


def expected_score(elo_diff: float) -> float:
    """Expected score of a player rated elo_diff above its opponent."""
    if elo_diff >= 0:
        return 1 / (1 + 10 ** (-elo_diff / 400))
    odds = 10 ** (elo_diff / 400)  # rearranged so that large negative gaps cannot overflow
    return odds / (1 + odds)


@dataclass
class FitResult:
    """Fitted ratings. se, diff and covariance are relative to the component's anchor and raise
    ValueError for players in different components, whose ratings are not comparable."""

    ratings: dict     # key -> Elo
    games: dict       # key -> games played, prior excluded
    points: dict      # key -> points scored (W + D/2), prior excluded
    component: dict   # key -> component index; 0 holds the anchor
    anchors: dict     # component index -> anchor key
    iterations: int   # Newton iterations of the slowest component
    _covariance: dict = field(default_factory=dict, repr=False, compare=False)  # component -> Elo^2 matrix
    _row: dict = field(default_factory=dict, repr=False, compare=False)  # key -> its row, None for anchors

    def covariance(self, a, b) -> float:
        """Covariance of the ratings of a and b in Elo^2; 0.0 if either is an anchor."""
        if self.component[a] != self.component[b]:
            raise ValueError(f"{a!r} and {b!r} are in different rating components and cannot be compared")
        i, j = self._row[a], self._row[b]
        if i is None or j is None:
            return 0.0
        return float(self._covariance[self.component[a]][i, j])

    def se(self, key) -> float:
        """Standard error (Elo) of key's rating; 0.0 for an anchor."""
        return math.sqrt(max(self.covariance(key, key), 0.0))

    def diff(self, a, b) -> tuple[float, float]:
        """(r_a - r_b, standard error of that difference)."""
        cross = self.covariance(a, b)
        variance = self.covariance(a, a) + self.covariance(b, b) - 2 * cross
        return self.ratings[a] - self.ratings[b], math.sqrt(max(variance, 0.0))


def fit_ratings(results, anchor=None, anchor_elo=1500.0, prior_draws=1.0, max_iter=100, tol=1e-6) -> FitResult:
    """Fit logistic Elo ratings to all results at once by maximum likelihood.

    results: iterable of (a, b, wins_a, draws, losses_a) tuples; keys are any hashable
    (ints or strings). Rows may repeat a pair in either orientation and are summed:
    (a, b, w, d, l) == (b, a, l, d, w). Rows with a == b or zero games are ignored, so
    only players with games are rated. Counts must be non-negative integers.

    anchor: key held at anchor_elo; its component is component 0. Every other component
    is held at anchor_elo by its smallest key. With anchor=None every component is
    anchored at its smallest key and component 0 is the largest. Remaining components
    are ordered by size, ties by smallest key (keys sort by type name, then value).
    prior_draws: virtual draws added to every pair that played each other (0: plain MLE).
    max_iter, tol: Newton iteration limit; converged when the largest step is below tol Elo.

    Raises ValueError for invalid counts or parameters, an anchor without games,
    infinite ratings (prior_draws=0 and a 100% score) and non-convergence. Input
    without any games gives an empty FitResult.
    """
    anchor_elo = float(anchor_elo)
    if not (math.isfinite(prior_draws) and prior_draws >= 0):
        raise ValueError(f"prior_draws must be finite and non-negative, got {prior_draws!r}")
    if not math.isfinite(anchor_elo) or max_iter < 1 or not tol > 0:
        raise ValueError("anchor_elo must be finite, max_iter at least 1 and tol positive")

    pairs = _aggregate(results)
    if not pairs:
        return FitResult({}, {}, {}, {}, {}, 0)
    players = sorted({key for pair in pairs for key in pair}, key=_order)
    index = {key: i for i, key in enumerate(players)}
    if anchor is not None and anchor not in index:
        raise ValueError(f"anchor {anchor!r} has not played any games")

    # Union-find components: the anchor's (or the largest) first, then by size and smallest key.
    parent = list(range(len(players)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[root(index[a])] = root(index[b])
    groups = {}
    for key in players:  # sorted, so every member list starts with its smallest key
        groups.setdefault(root(index[key]), []).append(key)
    anchor_root = None if anchor is None else root(index[anchor])
    components = [groups[r] for r in sorted(
        groups, key=lambda r: (r != anchor_root, -len(groups[r]), index[groups[r][0]]))]

    where = {key: (c, i) for c, members in enumerate(components) for i, key in enumerate(members)}
    scores = [np.zeros((len(members), len(members))) for members in components]  # [i, j]: i's points vs j
    for (a, b), (wins, draws, losses) in pairs.items():
        c, i = where[a]
        j = where[b][1]
        scores[c][i, j] = wins + (draws + prior_draws) / 2
        scores[c][j, i] = losses + (draws + prior_draws) / 2

    ratings, component, anchors, covariance, row = {}, {}, {}, {}, {}
    iterations = 0
    for c, members in enumerate(components):
        anchors[c] = anchor if c == 0 and anchor is not None else members[0]
        fixed = members.index(anchors[c])
        _require_finite(scores[c], members)
        theta, theta_covariance, used = _newton(scores[c], fixed, max_iter, tol)
        iterations = max(iterations, used)
        covariance[c] = theta_covariance * ELO_PER_LOGIT ** 2
        for i, key in enumerate(members):
            ratings[key] = anchor_elo + ELO_PER_LOGIT * float(theta[i])
            component[key] = c
            row[key] = None if i == fixed else i - (i > fixed)

    games, points = dict.fromkeys(ratings, 0), dict.fromkeys(ratings, 0.0)
    for (a, b), (wins, draws, losses) in pairs.items():
        games[a] += wins + draws + losses
        games[b] += wins + draws + losses
        points[a] += wins + draws / 2
        points[b] += losses + draws / 2
    return FitResult(ratings, games, points, component, anchors, iterations, covariance, row)


def _order(key):
    """Sort key for mixed keys: by type name, then value (ints numerically, then strings)."""
    return type(key).__name__, key


def _count(value):
    if not (value >= 0 and float(value).is_integer()):
        raise ValueError(f"game counts must be non-negative integers, got {value!r}")
    return int(value)


def _aggregate(results):
    """Sum rows per unordered pair: {(a, b): [wins_a, draws, losses_a]}, pairs with games only."""
    pairs = {}
    for a, b, wins, draws, losses in results:
        counts = [_count(wins), _count(draws), _count(losses)]
        if a == b or not sum(counts):
            continue
        if (b, a) in pairs:
            a, b, counts = b, a, counts[::-1]
        pairs[a, b] = [total + n for total, n in zip(pairs.get((a, b), (0, 0, 0)), counts)]
    return pairs


def _require_finite(scores, members):
    """Raise ValueError if a group scored 100% against the rest of its component.

    Finite maximum-likelihood ratings exist iff, however the component is split in
    two, each side scored against the other (Zermelo, Ford): the graph of "took
    points from" edges is strongly connected. Any positive prior guarantees this.
    """
    took_points = scores > 0  # [i, j]: i scored against j
    for edges, reached_won in ((took_points, False), (took_points.T, True)):
        reached = np.zeros(len(members), dtype=bool)
        reached[0] = True
        while True:
            grown = reached | edges[reached].any(axis=0)
            if (grown == reached).all():
                break
            reached = grown
        if not reached.all():
            # Closed under `edges`: reached took no points from the rest, or the rest none from reached.
            won = reached if reached_won else ~reached
            side, outcome = (won, "100%") if 2 * won.sum() <= len(members) else (~won, "0%")
            names = [members[i] for i in np.flatnonzero(side)]
            raise ValueError(f"ratings are infinite: {names} scored {outcome} against the rest of their "
                             "component; use prior_draws > 0")


def _newton(scores, anchor, max_iter, tol):
    """Maximize LL over theta (natural log-odds) with theta[anchor] = 0.

    Returns theta, the covariance of the other players' theta (inverse of the
    reduced negative Hessian) and the number of Newton iterations used.
    """
    free = np.arange(len(scores)) != anchor
    reduced = np.ix_(free, free)
    theta = np.zeros(len(scores))
    ll = _log_likelihood(scores, theta)
    for iteration in range(1, max_iter + 1):
        gradient, information = _gradient_information(scores, theta)
        step = np.zeros_like(theta)
        step[free] = np.linalg.solve(information[reduced], gradient[free])
        t = 1.0
        for _ in range(60):  # backtrack so that LL never decreases (beyond rounding)
            candidate = theta + t * step
            candidate_ll = _log_likelihood(scores, candidate)
            if candidate_ll >= ll - 1e-12 * abs(ll):
                break
            t /= 2
        else:
            raise ValueError("rating fit: the line search could not increase the likelihood")
        theta, ll = candidate, candidate_ll
        if np.abs(step).max() * ELO_PER_LOGIT < tol:
            break
    else:
        raise ValueError(f"rating fit did not converge in {max_iter} iterations "
                         f"(last step {np.abs(step).max() * ELO_PER_LOGIT:.3g} Elo)")
    covariance = np.linalg.inv(_gradient_information(scores, theta)[1][reduced])
    return theta, (covariance + covariance.T) / 2, iteration


def _log_likelihood(scores, theta):
    x = theta[:, None] - theta[None, :]
    return -float((scores * np.logaddexp(0.0, -x)).sum())  # sum of S_ij log p_ij over ordered pairs


def _gradient_information(scores, theta):
    """Gradient of LL and its negative Hessian (a weighted graph Laplacian)."""
    x = theta[:, None] - theta[None, :]
    p = np.exp(-np.logaddexp(0.0, -x))  # p[i, j]: expected score of i against j
    q = p.T  # 1 - p, without cancellation
    gradient = (scores * q - scores.T * p).sum(axis=1)  # sum_j S_ij - N_ij p_ij
    weights = (scores + scores.T) * p * q  # N_ij p_ij (1 - p_ij)
    return gradient, np.diag(weights.sum(axis=1)) - weights
