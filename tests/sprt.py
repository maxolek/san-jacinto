#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
from datetime import datetime
import sqlite3
from data import etl
import re
from pathlib import Path
import time
from datetime import datetime, timezone
import platform
import math
import scipy.stats as stats
import threading 
import queue

system = platform.system()

# paths
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
# log paths
LOGS_DIR = PROJECT_ROOT / "logs"
SPRT_LOG_DIR = LOGS_DIR / "sprt_logs"
GAME_JSON = SPRT_LOG_DIR / "game.jsonl"
SEARCH_JSON = SPRT_LOG_DIR / "search.jsonl"
TIMING_JSON = SPRT_LOG_DIR / "timing.jsonl"
ROOT_MOVES_JSON = SPRT_LOG_DIR / "root_moves.jsonl"

######################
#       MATH         
######################

def elo_to_score(elo):
    """convert elo diff to expected score"""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))

def score_to_elo(score):
    """convert score to elo diff"""
    if score <= 0:   return 0.0
    elif score >= 1: return 800.0
    else:            return -400.0 * math.log10(1.0 / score - 1.0)

def calculate_ci(variance, condifence_level=0.95):
    """calculate the CI for a given confidence level"""
    alpha = 1 - condifence_level
    tail_area = 1 - (alpha / 2)

    # dynamically find z-score (replaced hard coded 1.96 scalar for 95%)
    z_score = stats.norm.ppf(tail_area)
    ci_margin = z_score * math.sqrt(variance)
    return ci_margin

def compute_naive_elo(W, L):
    """naive elo ignoring draws entirely -- reference only, not decision-relevant"""
    if W == 0 or L == 0:
        return 0.0
    return 400.0 * math.log10(W / L)


def _bayes_elo_from_probs(p_win, p_loss):
    """compute BayesElo and drawElo from observed W/L probabilities."""
    bayes_elo = 200.0 * math.log10(
        p_win / p_loss * (1.0 - p_loss) / (1.0 - p_win)
    )
    draw_elo = 200.0 * math.log10(
        (1.0 - p_loss) / p_loss * (1.0 - p_win) / p_win
    )
    return bayes_elo, draw_elo 

def _scale(draw_elo):
    """BayesElo scale factor"""
    x = 10.0 ** (-draw_elo / 400.0)
    return 4.0 * x / ((1.0 + x) * (1.0 + x))

def _probs_from_bayes_elo(bayes_elo, draw_elo):
    """compute W/D/L probabilities from BayesElo parameters"""
    p_win = 1.0 / (1.0 + 10.0 ** ((draw_elo - bayes_elo) / 400.0))
    p_loss = 1.0 / (1.0 + 10.0 ** ((draw_elo + bayes_elo) / 400.0))
    p_draw = 1.0 - p_win - p_loss 
    return p_win, p_loss, p_draw 

def compute_llr(W, D, L, elo0, elo1):
    """
    compute log-likelihood ratio for SPRT using BayesElo logistic model
    
    1. regularize with jeffreys' prior (add .5 to each bin)
    2. estimate drawElo from the sample
    3. convert elo0/elo1 to BayesElo using the scale factor
    4. compute per-class log-likelihood ratio
    """
    if W <= 0 and L <= 0 and D <= 0: return 0.0

    # jeffreys prior regularization 
    wins = W + 0.5
    losses = L + 0.5
    draws = D + 0.5
    total = wins+losses+draws 

    p_win = wins/total 
    p_loss = losses/total 
    p_draw = 1.0 - p_win - p_loss 

    # estimate drawElo from observed data
    _, draw_elo = _bayes_elo_from_probs(p_win, p_loss)
    s = _scale(draw_elo) 

    # convert elo bounds to BayesElo space
    b0_elo = elo0 / s 
    b1_elo = elo1 / s 

    # probability models under bounds
    p0_win, p0_loss, p0_draw = _probs_from_bayes_elo(b0_elo, draw_elo)
    p1_win, p1_loss, p1_draw = _probs_from_bayes_elo(b1_elo, draw_elo)

    # avoid log(0)
    if p0_win <= 0 or p0_loss <= 0 or p0_draw <= 0: return 0.0
    if p1_win <= 0 or p1_loss <= 0 or p1_draw <= 0: return 0.0

    # log-likelihood ratio
    llr = (wins * math.log(p1_win / p0_win) +
            losses * math.log(p1_loss / p0_loss) +
            draws * math.log(p1_draw / p0_draw))
    
    return llr

def compute_elo_with_ci(W, D, L, CI=.95):
    """compute elo estimate and 95% confidence interval"""
    N = W + D + L 
    if N == 0: return 0.0, 0.0, 0.0

    w, d, l = W/N, D/N, L/N 
    s = w + d/2.0 
    elo = score_to_elo(s)

    # CI: via score variance
    m2 = w + d / 4.0
    var = (m2 - s**2) / N
    if var <= 0: return elo, elo, elo 

    ci = calculate_ci(var, CI)
    s_lo = max(0.001, s - ci)
    s_hi = min(0.999, s + ci)

    return elo, score_to_elo(s_lo), score_to_elo(s_hi)

def compute_bayes_elo_with_ci(W, D, L, CI=0.95):
    """
    BayesElo estimate + CI, using the same draw-adjusted scale as compute_llr.
    This is the number your SPRT pass/fail is actually computed against.
    """
    N = W + D + L
    if N == 0:
        return 0.0, 0.0, 0.0

    # jeffreys prior regularization (same as compute_llr, for consistency at low N)
    wins = W + 0.5
    losses = L + 0.5
    draws = D + 0.5
    total = wins + losses + draws

    p_win = wins / total
    p_loss = losses / total

    bayes_elo, draw_elo = _bayes_elo_from_probs(p_win, p_loss)
    s = _scale(draw_elo)

    # convert BayesElo back to "normalized" 400-scale elo for display
    elo = bayes_elo * s

    # CI via score variance, then map through the same scale
    score = p_win + draws / (2 * total)
    m2 = p_win + draws / (4 * total)
    var = (m2 - score ** 2) / total
    if var <= 0:
        return elo, elo, elo

    ci = calculate_ci(var, CI)
    s_lo = max(0.001, score - ci)
    s_hi = min(0.999, score + ci)

    elo_lo = score_to_elo(s_lo) * s
    elo_hi = score_to_elo(s_hi) * s

    return elo, elo_lo, elo_hi

def compute_normalized_elo_with_ci(W, D, L, CI=0.95):
    """
    Normalized Elo (nElo): rescales the naive logistic elo estimate by the
    ratio of a reference variance (0.25, i.e. no-draw 50/50 match) to the
    actual observed per-game score variance. Draws shrink variance, so nElo
    values run larger than regular Elo at high draw rates -- this is expected.
    """
    N = W + D + L
    if N == 0:
        return 0.0, 0.0, 0.0

    p_win, p_draw, p_loss = W / N, D / N, L / N
    mu = p_win + 0.5 * p_draw
    if mu <= 0.0 or mu >= 1.0:
        return 0.0, 0.0, 0.0

    elo = score_to_elo(mu)

    # empirical per-game variance of the score outcome (0 / 0.5 / 1)
    var = p_win * (1 - mu) ** 2 + p_draw * (0.5 - mu) ** 2 + p_loss * (0 - mu) ** 2
    if var <= 0:
        return elo, elo, elo

    ref_var = 0.25
    scale = math.sqrt(ref_var / var)
    nelo = elo * scale

    # CI on mu (score), same approach as compute_elo_with_ci, then rescaled
    m2 = p_win + p_draw / 4.0
    var_of_mean = (m2 - mu ** 2) / N
    if var_of_mean <= 0:
        return nelo, nelo, nelo

    ci = calculate_ci(var_of_mean, CI)
    mu_lo = max(0.001, mu - ci)
    mu_hi = min(0.999, mu + ci)

    nelo_lo = score_to_elo(mu_lo) * scale
    nelo_hi = score_to_elo(mu_hi) * scale

    return nelo, nelo_lo, nelo_hi

def compute_los_with_bounds(W, L, confidence=0.95):
    """
    CPW-style LOS with approximate confidence bounds.
    Uses decisive games only.
    """

    N = W + L
    if N == 0:
        return 50.0, 0.0, 100.0

    # CPW LOS
    z = (W - L) / math.sqrt(2.0 * N)
    los = 0.5 * (1 + math.erf(z / math.sqrt(2))) * 100

    # uncertainty in z
    # approximate variance of (W-L) is 2N
    # SE of numerator is sqrt(2N)
    # add/subtract one normal unit
    zc = 1.96

    z_low = z - zc
    z_high = z + zc

    def z_to_los(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2))) * 100

    return los, z_to_los(z_low), z_to_los(z_high)

###############################
#     LIVE PLOTTING
###############################

class LivePlotter:
    def __init__(self, 
                 elo0, elo1, alpha=0.05, beta=0.05, 
                 engine_a_data=None, engine_b_data=None, 
                 time=None, depth=None, tc=None
    ):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.candidate_data = engine_a_data
        self.baseline_data = engine_b_data
        self.elo0 = elo0 
        self.elo1 = elo1
        self.lbound = math.log(beta / (1 - alpha))
        self.ubound = math.log((1 - beta) / alpha)

        # time control
        if time: 
            self.tc = time * 1000 # time is given in seconds, want milliseconds
            self.tc_label = ' ms'
        elif depth: 
            self.tc = depth
            self.tc_label = ' depth' 
        elif tc: 
            self.tc = tc
            self.tc_label = ' ' # TC is explicit enough

        # data series
        self.games = []
        self.llr_series = []
        self.elo_series = []
        self.elo_lo_series = []
        self.elo_hi_series = []
        self.bayes_elo_series = []
        self.bayes_elo_lo_series = []
        self.bayes_elo_hi_series = []
        self.normalized_elo_series = []
        self.normalized_elo_lo_series = []
        self.normalized_elo_hi_series = []
        self.score_series = []

        # per color tracking (candidate perspective)
        self.white_w = 0
        self.white_l = 0
        self.white_d = 0
        self.black_w = 0
        self.black_l = 0
        self.black_d = 0

        # opening pair tracking: pairs are (game 1, game 2), (game 3, game 4), etc.
        self.pair_ww = 0 # candidate wins both
        self.pair_ll = 0 # candidate losses both
        self.pair_dd = 0 # candidate draws both
        self.pair_wl = 0 # candidate wins one loses another
        self.pair_wd = 0 # 1 W - 1 D
        self.pair_ld = 0 # 1 L - 1 D
        self.pending_pair_result = None # result of first game in pair (but dont have both yet)
        self.pair_results = {}

        # setup figure: 2x2 grid (LLR, Elo, Score, Stats)
        self.plt.ion()
        self.fig = self.plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                                   left=0.06, right=0.98, top=0.92, bottom=0.08)
        self.ax_llr = self.fig.add_subplot(gs[0,0])
        self.ax_elo = self.fig.add_subplot(gs[0,1])
        self.ax_score = self.fig.add_subplot(gs[1,0])
        self.ax_stats = self.fig.add_subplot(gs[1,1])
        self.fig.suptitle(f"SPRT Live Monitor    {self.tc}{self.tc_label}", fontweight='bold', fontsize=11)

        # LLR plot (top-left)
        self.ax_llr.set_title('LLR')
        self.ax_llr.set_ylabel("LLR")
        self.ax_llr.set_xlabel('Games')
        self.ax_llr.axhline(self.ubound, color='green', linestyle='--', alpha=0.7, label=f'H1 ({self.ubound:.2f})')
        self.ax_llr.axhline(self.lbound, color='red', linestyle='--', alpha=0.7, label=f'H0 ({self.lbound:.2f})')
        self.ax_llr.axhline(0, color='grey', linestyle='-', alpha=0.3)
        self.ax_llr.legend(loc='upper left', fontsize=7)
        self.line_llr, = self.ax_llr.plot([], [], 'b-', linewidth=1.5)
        self.ax_llr.set_ylim(self.lbound * 1.05, self.ubound * 1.05)

        # elo plot (top-right)
        self.ax_elo.set_title('ELO')
        self.ax_elo.set_xlabel('Games')
        self.ax_elo.set_ylabel('Elo')
        self.ax_elo.axhline(0, color='grey', linestyle=':', alpha=0.3)
        self.ax_elo.axhline(self.elo0, color='red', linestyle=':', alpha=0.5, label=f'elo0={self.elo0}')
        self.ax_elo.axhline(self.elo1, color='green', linestyle=':', alpha=0.5, label=f'elo1={self.elo1}')
        # BayesElo 
        self.line_bayes_elo, = self.ax_elo.plot([], [], color='black', linewidth=1.5, label='BayesElo')
        self.fill_bayes_elo = None
        # normalized
        self.line_nelo, = self.ax_elo.plot([], [], color='purple', linewidth=1.0, alpha=1.0,label='Normalized (nElo)')
        self.fill_nelo = None
        # naive
        self.line_elo, = self.ax_elo.plot([], [], color='orange', linewidth=1.0, alpha=1.0, label='Logistic (Elo)')
        self.fill_elo = None
        self.ax_elo.legend(loc='upper left', fontsize=7)
        self.ax_elo.set_ylim(-100,100)

        # score plot (bottom-left)
        self.ax_score.set_title('SCORE')
        self.ax_score.set_xlabel('Games')
        self.ax_score.set_ylabel('Score')
        self.ax_score.axhline(0.5,  color='grey', linestyle='-', alpha=0.3)
        self.s0 = elo_to_score(elo0)
        self.s1 = elo_to_score(elo1)
        self.ax_score.axhline(self.s0, color='red', linestyle=':', alpha=0.5, label=f"s(elo0)={self.s0:.3f}")
        self.ax_score.axhline(self.s1, color='green', linestyle=':', alpha=0.5, label=f"s(elo1)={self.s1:.3f}")
        self.ax_score.legend(loc='upper left', fontsize=7)
        self.line_score, = self.ax_score.plot([], [], 'b-', linewidth=1.5)
        self.ax_score.set_ylim(.3, .7)

        # stats text panel (bottom-right))
        self.ax_stats.set_title('STATS')
        self.ax_stats.set_xticks([])
        self.ax_stats.set_yticks([])
        self.ax_stats.set_facecolor("#f8f8f8")
        for spine in self.ax_stats.spines.values():
            spine.set_color("#cccccc")
        self.stats_text = self.ax_stats.text(
            0.05, 0.95, '', fontsize=10, fontfamily='monospace',
            verticalalignment='top', transform=self.ax_stats.transAxes
        )

        #self.plt.tight_layout(rect=[0, 0, 0.62, 0.95])
        self.plt.show(block=False)
        self.fig.canvas.draw()
        # grab the Tk root for direct event pumping (avoids TkAgg blocking issues)
        self._tk_root = None 
        try:
            self._tk_root = self.fig.canvas.manager.window 
        except Exception:
            pass

    def add_game(self, game_num, candidate_is_white, result_str):
        """track a finished game result. result_str is e.g. '1-0', '0-1', '1/2-1/2' """
        # determine outcome from candidate perspective
        if result_str == '1-0':
            cand_result = 'w' if candidate_is_white else 'l'
        elif result_str == '0-1':
            cand_result = 'l' if candidate_is_white else 'w'
        else:
            cand_result = 'd'

        # per color stats
        if candidate_is_white:
            if cand_result == 'w': self.white_w += 1
            elif cand_result == 'l': self.white_l += 1 
            else: self.white_d += 1
        else:
            if cand_result == 'w': self.black_w += 1
            elif cand_result == 'l': self.black_l += 1 
            else: self.black_d += 1

        # opening pair tracking (games 1&2 = pair0, games 3&4 = pair1, etc.)
        pair_id = (game_num - 1) // 2
        self.pair_results.setdefault(pair_id, []).append(cand_result)
        if len(self.pair_results[pair_id]) == 2:
            pair = ''.join(sorted(self.pair_results.pop(pair_id)))
            if pair == 'ww': self.pair_ww += 1
            elif pair == 'll': self.pair_ll +=1 
            elif pair == 'dd': self.pair_dd +=1 
            elif pair == 'lw': self.pair_wl +=1 
            elif pair == 'dw': self.pair_wd +=1 
            elif pair == 'dl': self.pair_ld +=1 

    def update(self, W, D, L):
        N = W + D + L 
        if N == 0: return 

        self.games.append(N)

        # LLR
        llr = compute_llr(W, D, L, self.elo0, self.elo1)
        self.llr_series.append(llr) 

        # elo + ci
        elo, elo_lo, elo_hi = compute_elo_with_ci(W, D, L)
        b_elo, b_elo_lo, b_elo_hi = compute_bayes_elo_with_ci(W, D, L)
        n_elo, n_elo_lo, n_elo_hi = compute_normalized_elo_with_ci(W, D, L)

        self.normalized_elo_series.append(n_elo)
        self.bayes_elo_series.append(b_elo)
        self.elo_series.append(elo)

        self.normalized_elo_lo_series.append(n_elo_lo)
        self.bayes_elo_lo_series.append(b_elo_lo)
        self.elo_lo_series.append(elo_lo)

        self.normalized_elo_hi_series.append(n_elo_hi)
        self.bayes_elo_hi_series.append(b_elo_hi)
        self.elo_hi_series.append(elo_hi)

        # update LLR line
        self.line_llr.set_data(self.games, self.llr_series)
        self.ax_llr.set_xlim(1, max(self.games) * 1.1)
        llr_lo = min(self.llr_series) - 0.2
        llr_hi = max(self.llr_series) + 0.2
        self.ax_llr.set_ylim(min(self.lbound * 1.1, llr_lo), max(self.ubound * 1.1, llr_hi))

        # update elo line
        self.line_elo.set_data(self.games, self.elo_series)
        self.line_bayes_elo.set_data(self.games, self.bayes_elo_series)
        self.line_nelo.set_data(self.games, self.normalized_elo_series)
        if self.fill_elo:
            self.fill_elo.remove()
            self.fill_bayes_elo.remove()
            self.fill_nelo.remove()
        self.fill_elo = self.ax_elo.fill_between(
            self.games, self.elo_lo_series, self.elo_hi_series, alpha=0.4, color='orange'
        )
        self.fill_nelo = self.ax_elo.fill_between(
            self.games, self.normalized_elo_lo_series, self.normalized_elo_hi_series, alpha=0.2, color='purple'
        )
        self.fill_bayes_elo = self.ax_elo.fill_between(
            self.games, self.bayes_elo_lo_series, self.bayes_elo_hi_series, alpha=0.2, color='blue', edgecolor='black'
        )
        self.ax_elo.relim()
        self.ax_elo.autoscale_view()
        # clamp elo y-axis to reasonable range (extreme early, smoothes and converges later)
        # skip games 1+2 -- elo estimate starts at 0 or +/-800 and isn't representative
        lo_tail = self.elo_lo_series[2:]
        hi_tail = self.elo_hi_series[2:]
        n = len(lo_tail)
        if n == 0:
            # only game 1+2 exists -- fall back to full series so we don't crash
            lo_tail = self.elo_lo_series
            hi_tail = self.elo_hi_series
            n = len(lo_tail)
        if n < 20:
            elo_min = min(lo_tail)
            elo_max = max(hi_tail)
        else:
            cut = 2 * n // 3
            elo_min = min(lo_tail[cut:])
            elo_max = max(hi_tail[cut:])
        elo_pad = max(5, (elo_max - elo_min) * 0.05)
        plot_elo_lo = min(
            min(self.elo0, self.elo1) - 5,
            -5, 
            elo_min - elo_pad
        )
        plot_elo_hi = max(
            max(self.elo0, self.elo1) + 5,
            5, 
            elo_max + elo_pad
        )
        self.ax_elo.set_ylim(plot_elo_lo, plot_elo_hi)

        # title with current stats
        score = (W + D/2.0) / N 
        self.score_series.append(score)
        los, los_lo, los_hi = compute_los_with_bounds(W, L)
        self.fig.suptitle(
            f"SPRT Live | {self.tc}{self.tc_label} | Games: {N} | Score: {score:.3f} | "
            f"Elo: {elo:+.1f} ({elo_lo:+.1f}, {elo_hi:+.1f}) | LOS: {los:.1f}% ({los_lo:.1f}, {los_hi:.1f}) | LLR: {llr:.2f}",
            fontweight='bold', fontsize=12
        )

        #  update score line
        self.line_score.set_data(self.games, self.score_series)
        self.ax_score.set_xlim(1, max(self.games) * 1.05)
        # skip games 1+2 -- score estimate is noisy/unrepresentative this early
        score_tail = self.score_series[2:]
        n = len(score_tail)
        if n == 0:
            # only games 1-2 exist -- fall back to full series so we don't crash
            score_tail = self.score_series
            n = len(score_tail)
        if n < 30:
            s_min = min(score_tail)
            s_max = max(score_tail)
        else:
            cut = n // 5 # cut out noise start (0<->1 oscillations)
            s_min = min(score_tail[cut:])
            s_max = max(score_tail[cut:])
        s_pad = max(0.02, (s_max - s_min) * 0.5)
        s_lo, s_hi = min(min(self.s0,self.s1), s_min), max(max(self.s0,self.s1), s_max)
        self.ax_score.set_ylim(s_lo - s_pad, s_hi + s_pad)

        # stats text panel
        wt = self.white_w + self.white_l + self.white_d 
        bt = self.black_w + self.black_l + self.black_d 
        pairs = self.pair_ww + self.pair_ll + self.pair_dd + self.pair_wl + self.pair_wd + self.pair_ld

        lines = []
        lines.append(f"-- Overall --             Candidate Version: {self.candidate_data['version']}")
        lines.append(f"  W-D-L: {W}-{D}-{L}         Baseline Version: {self.baseline_data['version']}") #\t\t{self.candidate_data['description']}
        lines.append("")

        lines.append("-- As White --")
        if wt > 0:
            w_score = (self.white_w + self.white_d / 2.0) / wt 
            lines.append(f"  W-D-L: {self.white_w}-{self.white_d}-{self.white_l}")
            lines.append(f"  Score: {w_score:.3f}")
        else:
            lines.append("  (no games)")
        lines.append("")

        lines.append(f"-- As Black --            Elo: {elo:+.1f} ({elo_lo:+.1f}, {elo_hi:+.1f})")
        if bt > 0:
            b_score = (self.black_w + self.black_d / 2.0) / bt 
            lines.append(f"  W-D-L: {self.black_w}-{self.black_d}-{self.black_l}            nElo: {n_elo:+.1f} ({n_elo_lo:+.1f}, {n_elo_hi:+.1f})") #\t\t{self.baseline_data['description']}
            lines.append(f"  Score: {b_score:.3f}         BayesElo: {b_elo:+.1f} ({b_elo_lo:+.1f}, {b_elo_hi:+.1f})")
        else:
            lines.append("  (no games)")
        lines.append("")

        lines.append("-- Opening Pairs --")
        if pairs > 0:
            lines.append(f"  WW: {self.pair_ww}  DD: {self.pair_dd}  LL: {self.pair_ll}")
            lines.append(f"  WD: {self.pair_wd}  WL: {self.pair_wl}  DL: {self.pair_ld}")
            pair_score = (2*self.pair_ww + 1.5*self.pair_wd + self.pair_wl + self.pair_dd + .5*self.pair_ld) / (2*pairs)
            lines.append(f"  Pair score: {pair_score:.3f}")
        else:
            lines.append("  (no pairs)")

        self.stats_text.set_text('\n'.join(lines))

        self.fig.canvas.draw_idle()
        self._pump_events()

    def finalize(self, result):
        color = {'pass': 'green', 'fail': 'red'}.get(result, 'yellow')
        self.ax_llr.set_title(f"Result: {result.upper()}", color=color, fontweight='bold')
        
        # --- expand all axes to show the full history, not just the last tail ---
        # x-axis: full game range on both time-series plots
        self.ax_llr.set_xlim(1, max(self.games) * 1.05)
        self.ax_score.set_xlim(1, max(self.games) * 1.05)

        # LLR: full min/max over the whole series (same logic as update(), just untailed)
        llr_lo = min(self.llr_series) - 0.2
        llr_hi = max(self.llr_series) + 0.2
        self.ax_llr.set_ylim(min(self.lbound * 1.1, llr_lo), max(self.ubound * 1.1, llr_hi))

        # Elo: use full lo/hi series (skip games 1+2 as before, but no tail cut)
        lo_tail = self.elo_lo_series[2:] or self.elo_lo_series
        hi_tail = self.elo_hi_series[2:] or self.elo_hi_series
        elo_min = min(lo_tail)
        elo_max = max(hi_tail)
        elo_pad = max(5, (elo_max - elo_min) * 0.05)
        plot_elo_lo = min(min(self.elo0, self.elo1) - 5, -5, elo_min - elo_pad)
        plot_elo_hi = max(max(self.elo0, self.elo1) + 5, 5, elo_max + elo_pad)
        self.ax_elo.set_ylim(plot_elo_lo, plot_elo_hi)

        # Score: use full series (skip games 1+2 as before, but no tail cut)
        score_tail = self.score_series[2:] or self.score_series
        s_min = min(score_tail)
        s_max = max(score_tail)
        s_lo, s_hi = min(0.495, s_min), max(0.505, s_max)
        s_pad = max(0.005, (s_max - s_min) * 0.05)
        self.ax_score.set_ylim(s_lo - s_pad, s_hi + s_pad)

        self.fig.canvas.draw_idle()
        self.plt.ioff()
        self.plt.show()

    def _pump_events(self):
        """non-blocking GUI event pump"""
        try:
            if self._tk_root is not None: 
                self._tk_root.update()
            else:
                self.fig.canvas.flush_events()
        except Exception:
            pass

##########################
#  RUN SPRT
##########################

def parse_cutechess_output(output, candidate_name="Candidate"):

    def safe_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    def safe_float(x):
        try:
            v = float(x)
            if not (v == v and abs(v) != float("inf")):
                return None
            return v
        except Exception:
            return None

    stats = {
        "candidate_wins": 0,
        "candidate_losses": 0,
        "candidate_draws": 0,

        "candidate_white_wins": 0,
        "candidate_white_losses": 0,
        "candidate_white_draws": 0,

        "candidate_black_wins": 0,
        "candidate_black_losses": 0,
        "candidate_black_draws": 0,

        "games_played": 0,
        "result": None,
    }

    # --- Regexes ---
    score_re = re.compile(
        rf"Score of {re.escape(candidate_name)} vs .+?:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
    )

    white_re = re.compile(
        rf"\.\.\.\s*{re.escape(candidate_name)} playing White:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
    )

    black_re = re.compile(
        rf"\.\.\.\s*{re.escape(candidate_name)} playing Black:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
    )

    number = r"[+-]?(?:\d+(?:\.\d+)?|inf|nan)"
    elo_re = re.compile(
        rf"""
        Elo\ difference:\s*(?P<elo>{number})\s*\+/-\s*{number}
        ,\s*LOS:\s*(?P<los>{number})\s*%
        ,\s*DrawRatio:\s*(?P<draw>{number})\s*%
        """,
        re.IGNORECASE | re.VERBOSE
    )

    sprt_re = re.compile(
        r"SPRT:\s*llr\s*([-\d\.]+)\s*\([^)]+\),\s*lbound\s*([-\d\.]+),\s*ubound\s*([-\d\.]+)",
        re.IGNORECASE,
    )

    # We overwrite on every match → last one wins
    for line in output.splitlines():

        line = line.strip()

        if m := score_re.search(line):
            w, l, d = map(safe_int, m.groups())
            stats["candidate_wins"] = w
            stats["candidate_losses"] = l
            stats["candidate_draws"] = d
            stats["games_played"] = w + l + d
            continue

        if m := white_re.search(line):
            stats["candidate_white_wins"] = safe_int(m.group(1))
            stats["candidate_white_losses"] = safe_int(m.group(2))
            stats["candidate_white_draws"] = safe_int(m.group(3))
            continue

        if m := black_re.search(line):
            stats["candidate_black_wins"] = safe_int(m.group(1))
            stats["candidate_black_losses"] = safe_int(m.group(2))
            stats["candidate_black_draws"] = safe_int(m.group(3))
            continue

        if m := elo_re.search(line):
            stats["elo_diff"] = safe_float(m.group("elo"))
            stats["los"] = safe_float(m.group("los"))
            stats["draw_ratio"] = safe_float(m.group("draw"))
            continue

        if m := sprt_re.search(line):
            stats["llr"] = safe_float(m.group(1))
            stats["lbound"] = safe_float(m.group(2))
            stats["ubound"] = safe_float(m.group(3))
            continue

    # derive final result string
    #   improvement test: elo1 > elo0 
    #   non-regression test: elo1 < elo0
    #       [BOTH] llr > ubound = pass, llr < lbound = fail, else inconclusive
    if stats["llr"] > stats["ubound"]:
        stats["result"] = "pass"
    elif stats["llr"] < stats["lbound"]:
        stats["result"] = "fail"
    else:
        stats["result"] = "inconclusive"

    return stats



def upload_logs(args, cute_chess_stats, runtime=None):
    print("[DATA] Preparing to upload logs to database...")

    if system == "Windows": cnxn = sqlite3.connect('F:/databases/chess.db')
    elif system == "Darwin": cnxn = sqlite3.connect(Path.home() / "Documents/databases/chess.db")

    print("[DATA] Probing engine metadata...")
    candidate_engine_version = etl.probe_engine_metadata(args.engine_a)['version']
    baseline_engine_version = etl.probe_engine_metadata(args.engine_b)['version']

    # get engine_id by probing db.engines via version
    print("[DATA] Retrieving engine ids...")
    candidate_engine_id = etl.get_engine_id(cnxn, version=candidate_engine_version)
    baseline_engine_id = etl.get_engine_id(cnxn, version=baseline_engine_version)

    # auto-register if not found
    if candidate_engine_id is None:
        print(f"[SPRT] Candidate engine {candidate_engine_version} not registered, registering now...")
        candidate_engine_id = etl.register_engine(cnxn, {"engine_path": args.engine_a})
    if baseline_engine_id is None:
        print(f"[SPRT] Baseline engine {baseline_engine_version} not registered, registering now...")
        baseline_engine_id = etl.register_engine(cnxn, {"engine_path": args.engine_b})

    # log sprt experiment
    sprt_id = etl.start_experiment(
        cnxn, 
        "SPRT",
        candidate_engine_id,
        comparison_engine_id = baseline_engine_id
    )

    # Only consolidate and ingest JSONL data if logging was enabled
    ingestion_ok = False
    if args.log:
        # consolidate per-instance log files from concurrent engine processes
        print("[DATA] Consolidating per-instance log files...")
        etl.consolidate_instance_logs(args.logroot)

        try:
            # map search --> game
            print("[DATA] Building game map ...")
            game_map = etl.bulk_log_game(
                cnxn, 
                GAME_JSON, 
                sprt_id,
            )

            # log search+timing with game mapping
            print("[DATA] Logging all search data ...")
            etl.bulk_log_search_and_timing(
                cnxn, 
                SEARCH_JSON,
                game_map, 
                timing_path=TIMING_JSON,
                root_moves_path=ROOT_MOVES_JSON
            )
            ingestion_ok = True
        except Exception as e:
            print(f"[DATA] JSONL ingestion failed: {e}")
            print("[DATA] Log files preserved for retry.")
    else:
        print("[DATA] Logging was disabled, skipping JSONL ingestion.")

    # log sprt experiment details in db.sprt (always, regardless of --log)
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    etl.log_sprt(
        cnxn,
        sprt_id,  # experiment_id
        candidate_engine_id,
        baseline_engine_id,
        **{**args_dict, **cute_chess_stats},
        runtime=runtime
    )
    etl.update_experiment(
        cnxn, 
        sprt_id, 
        {"end_time_utc": datetime.now(timezone.utc).isoformat()}
    )

    # Only clear log directory if ingestion succeeded
    if args.log and ingestion_ok:
        print("[DATA] Clearing log directory...")
        etl.clear_log_dir(args.logroot)
        print(f"[DATA] Logging completed for SPRT {sprt_id}.")


def parse_args():
    p = argparse.ArgumentParser(description="SPRT runner using cutechess-cli")

    # Engines
    p.add_argument("--engine-a", required=True, help="Candidate engine path")
    p.add_argument("--engine-b", required=True, help="Baseline engine path")
    p.add_argument("--params", type=dict, help="Dictionary of parameter and their values to test against the current version of the engine")

    # cutechess
    p.add_argument(
        "--cutechess-cli",
        default=r"C:\Program Files (x86)\Cute Chess\cutechess-cli.exe",
        help="Path to cutechess-cli.exe"
    )

    # Time control (choose ONE)
    p.add_argument("--depth", type=int, default=None, help="Depth per move")
    p.add_argument("--time", type=float, default=None, help="Seconds per move")
    p.add_argument("--tc", type=str, default=None, help="Time control (e.g. 0+1)")

    # SPRT parameters
    p.add_argument("--elo0", type=int, default=0)
    p.add_argument("--elo1", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--max-games", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=2, help="Number of concurrent games")

    # Opening book
    p.add_argument("--book", default= PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn" , help="Opening book file")
    p.add_argument("--book-depth", type=int, default=16) # 8 full moves

    # Logging
    p.add_argument('--log', action="store_true", help="Flag to turn on logging for candidate engine")
    p.add_argument('--plot', action='store_true', help='Show live SPRT plots (LLR, Elo)')
    p.add_argument(
        "--logroot",
        default=SPRT_LOG_DIR,
        help="Root directory for SPRT logs"
    )

    return p.parse_args()



def main(args=None):
    if args is None:
        args = parse_args()

    # Validate TC
    if sum(x is not None for x in (args.depth, args.time, args.tc)) != 1:
        raise ValueError("Specify exactly one of --depth, --time, or --tc")

    print(f"[SPRT] Log directory: {args.logroot}")

    engine_a = os.path.abspath(args.engine_a)
    engine_b = os.path.abspath(args.engine_b)

    """
    these logging controls are now moot with with dual dev/prod binary release
    + dev is always log (only exception is toggle of uci) and prod is never log
    """

    if args.time is not None:
        fast_tc = args.time <= 0.5
    else:  # args.tc must exist
        base_sec = 60*int(re.split(":", args.tc)[0]) + int(re.split(":", args.tc)[1][:2])
        fast_tc = base_sec <= 30
    should_log = args.log #(args.log and fast_tc) 

    if (should_log):
        log_reason = f"(low time control ... logging can affect performance at these search speeds)" if fast_tc else "Candidate logging on: baseline logging on for reduced change variables"
        print(f"[SPRT] Logging enabled for candidate engine {log_reason}")

    each_block = [
        "-each",
        "proto=uci"
    ]
    log_a_block = [
        f"option.log_dir={args.logroot}",
        f"option.uci_logging=true",
    ]
    for name, value in getattr(args, "params", {}): # add tuning result param values
        log_a_block.append(
            f"option.{name}={value}"
        )
    log_b_block = [
        f"option.log_dir={args.logroot}",
        f"option.uci_logging={"true" if should_log else "false"}",
    ]

    # Time control
    if args.depth is not None:
        each_block.append(f"depth={args.depth}")
    elif args.time is not None:
        each_block += [f"st={args.time}", "timemargin=30"]
    else:
        each_block.append(f"tc={args.tc}")

    # opening book
    book_block = []
    if args.book is not None:
        book_block.append("-openings")
        book_block.append(f"file={os.path.abspath(args.book)}")
        book_block.append(f"format={os.path.splitext(args.book)[1][1:]}")
        book_block.append("order=random")
        book_block.append(f"plies={args.book_depth}")
        #book_block.append("-srand")
        #book_block.append("42")

    cmd = [
        args.cutechess_cli,

        # Candidate engine
        "-engine",
        "name=Candidate",
        f"cmd={engine_a}",
        #"option.stats_nodes_only=true",
        f"dir={os.path.dirname(engine_a)}",
    ] + log_a_block + [

        # Baseline engine
        "-engine",
        "name=Baseline",
        f"cmd={engine_b}",
        f"dir={os.path.dirname(engine_b)}",
    ] + log_b_block + each_block + [

        # SPRT
        "-maxmoves", "100",
        "-games", str(args.max_games),
        "-sprt",
        f"elo0={args.elo0}",
        f"elo1={args.elo1}",
        f"alpha={args.alpha}",
        f"beta={args.beta}",
    ] + book_block + [

        # Runtime
        "-repeat",
        "-concurrency", str(args.concurrency),
        "-pgnout", os.path.join(args.logroot, "cc_sprt.pgn"),
    ]

    print("[SPRT] Launching cutechess:")
    print(" ".join(cmd))

    output_lines = []
    start_time = time.time()

    stdout_log_path = Path(args.logroot) / "cutechess_stdout.log"
    stderr_log_path = Path(args.logroot) / "cutechess_stderr.log"

    #stdout_f = open(stdout_log_path, "w", encoding="utf-8")
    #stderr_f = open(stderr_log_path, "w", encoding="utf-8")

    # probe engine data (for plotting)
    candidate_engine_data = etl.probe_engine_metadata(args.engine_a)
    baseline_engine_data = etl.probe_engine_metadata(args.engine_b)

    # Live plotter (init before Popen to its in scope after)
    plotter = None 
    if args.plot:
        plotter = LivePlotter(args.elo0, args.elo1, args.alpha, args.beta, candidate_engine_data, baseline_engine_data, args.time, args.depth, args.tc)

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        #stderr=subprocess.PIPE,   # IMPORTANT: do NOT merge streams
        text=True,
        bufsize=1  # line-buffered
    ) as proc:
        
        # live parsing regexes
        score_re = re.compile(
            r"Score of Candidate vs .+?:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\[([0-9.]+)\]\s*(\d+)"
        )
        finished_re = re.compile(
            r"Finished game (\d+) \((.+?) vs (.+?)\):\s*(\S+)"
        )
        
        # read stdout in a thread so main thread can pump GUI events
        line_queue = queue.Queue()

        def _reader():
            for ln in proc.stdout:
                line_queue.put(ln)
            line_queue.put(None) # sentinel
    
        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        while True:
            # drain all available lines (non-blocking)
            done = False
            while True:
                try:
                    line = line_queue.get_nowait()
                except queue.Empty:
                    break 
                if line is None: # EOF sentinel
                    done = True
                    break 
                print(line, end="")
                output_lines.append(line)

                if plotter:
                    # parse finished game lines for per-color and opening-pair stats
                    m = finished_re.search(line)
                    if m:
                        game_num = int(m.group(1))
                        first_player = m.group(2) 
                        result_str = m.group(4) 
                        candidate_is_white = (first_player == "Candidate")
                        plotter.add_game(game_num, candidate_is_white, result_str)

                    # parse score updates for live plotting
                    m = score_re.search(line) 
                    if m:
                        W, L, D = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        plotter.update(W, D, L)

            # pump GUI events to keep window responsive
            if plotter:
                plotter._pump_events()

            if done: 
                break

            # brief sleep to avoid busy-wait
            time.sleep(0.05)

        ret = proc.wait()

    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)

    run_time = time.time() - start_time
    output = "".join(output_lines)
    stats = parse_cutechess_output(output)

    if plotter:
        plotter.finalize(stats.get('result', 'inconclusive'))

    upload_logs(args, cute_chess_stats=stats, runtime=run_time)

    print("[SPRT] Completed successfully")

    return {
        "accepted": stats['result'] == "pass",
        "elo": stats['elo_diff'],
        "games": stats['games_played']
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[SPRT] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
