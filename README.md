# Blackjack

A from-scratch blackjack bot for a six-deck shoe. It plays the hand, sizes the bet, and decides insurance from the cards still left in the shoe. Nothing here is taken from an existing blackjack library. The engine, the exact-probability baseline, the networks, and the statistics are all in this repository, and the only dependency is NumPy.

The bot has to bet at least 1 unit on every hand. On a holdout of 1,000,000 hands it made **+0.03049 per hand**, against **+0.01606** for full-Kelly play from the infinite-deck probability calculation on the same shoes. The paired difference is **+0.01443** (95% CI +0.00963 to +0.01923).

That profit is not a 3% edge on a flat bet. The average stake was 2.59 units, so the return on money actually wagered is about 1.2%. Most of it comes from betting more when the remaining shoe favors the player.

## Rules

- 6 decks, reshuffled after 75% of the cards are dealt
- Dealer stands on soft 17
- Blackjack pays 3:2
- Double on any first two cards, including after a split
- Late surrender
- Split up to four hands (three splits)
- No hitting or resplitting aces
- Ten-value cards pair with each other
- Insurance is offered on a dealer ace, costs half the bet, and pays 2:1

The dealer’s hole card is hidden from the player until the hand is over, and it is counted as soon as it is revealed.

## How it was built

The project started as an attempt to learn the game by reinforcement learning, with an exact probability engine kept alongside it as a measuring stick.

**The shoe and the rules** live in `blackjack_solver/engine.py`. A shoe tracks the remaining ranks and a Hi-Lo running count. `play_hand` deals, offers insurance, resolves splits, doubles, and surrender, then pays the hand. Player decisions never see the hole card.

**The probability baseline** is in `blackjack_solver/exact.py`. It is backward induction on an infinite shoe: ace through 9 at 1/13, tens at 4/13, dealer stands on soft 17. It is derived from these rules, not copied from a published basic-strategy chart. Under those probabilities the value of flat basic strategy is about **−0.43%** per hand. The calculation is optimal only for an infinite shoe. It ignores which specific cards have already been dealt.

**Reinforcement learning did not reach that strategy.** One-step Q-learning, on a table and then on a small NumPy network, improved a lot and then stopped. An untrained policy that mostly hits is about **−0.64** per hand. After 80,000 hands the learned policy was still about **−0.085** per hand, and it agreed with the probability baseline on only about 58% of decisions. Probes of famous spots (stand on 20, double 11, split aces) could look right while the rest of the policy stayed several points worse than basic strategy. The full write-up of that run is `results/report.txt`.

**Play was then taught by imitation, conditioned on the shoe.** `blackjack_solver/composition.py` reruns the same backward induction with the shoe’s current rank probabilities instead of the infinite-deck mix. On close decisions, and when the shoe is unbalanced, that action differs from the infinite-deck chart. `blackjack_solver/same_holdout.py` collects those actions and trains a two-layer network by cross-entropy. The true-count input is zeroed, so the network has to read the remaining ranks themselves.

On 2,000,000 identical hands, both betting 1 unit every hand, that network beat infinite-deck play by **+0.00128** per hand (95% CI +0.00090 to +0.00165). Both still lost to the house: the network at −0.00328, the probability chart at −0.00456. Better play alone does not overcome the house edge.

**Betting is a second network.** `blackjack_solver/internal.py` deals hundreds of thousands of flat-bet hands with the play network and fits the expected return from 11 shoe features: the fraction of each rank still unseen, and how much of the shoe is left. A linear least-squares layer carries most of that fit. A small residual network (11 → 32 → 32 → 1) is added on top and kept only when it lowers validation error; in practice the extra error reduction is tiny, so the stake is essentially a learned linear function of the remaining composition. Insurance is a network of the same shape, trained to predict whether a dealer blackjack is more likely than the break-even rate of 1 in 3.

The stake is full Kelly on a reference bankroll of 800, using the network’s predicted edge and the residual variance of a hand (at least 1). An earlier version bet 0 when the prediction was negative. That is profitable in the simulation, and it is not something a lone player can do at a table: the cards were still dealt, and a skipped hand was scored as zero. Casinos generally require a bet on every hand, and they often stop players who step in and out of a shoe. The current bot therefore bets **at least 1** on every hand and raises only when its own edge estimate is positive.

## How the bot works

Three networks see every hand. None of them is handed a Hi-Lo true count at decision time. The play network’s count feature is set to zero. The bet and insurance networks receive only the remaining rank fractions and the shoe depth.

**Bet, before the deal.** The value network predicts the expected profit of a 1-unit bet. The stake is

```text
min(22.2, max(1, 800 * max(predicted edge, 0) / variance))
```

A prediction of zero or negative stays at the minimum of 1. A prediction of about +1% becomes a bet of several units. On the 1,000,000-hand always-bet holdout the average stake was 2.59, and the cap of 22.2 was the 99th percentile of the full-Kelly stake on a calibration shoe, chosen because a hard cap of 12 was clipping the bets the model actually wanted.

**Play, after the cards are dealt.** A network with widths 64 → 128 → 128 → 5 chooses among hit, stand, double, split, and surrender. Illegal actions are masked. It was trained to copy the composition calculation, so the visible strategy is basic strategy plus the deviations that calculation makes when the shoe is short of some ranks and rich in others. On a neutral shoe it stands on 20, hits a hard 16 against a 10 only when that is correct, doubles 11, splits aces, and does not split tens.

**Insurance, when the dealer shows an ace.** A third network of the same family outputs a score trained against “probability of dealer blackjack, minus 1/3.” A positive score takes insurance. The break-even point is 1 in 3 because insurance pays 2:1 on a stake of half the bet.

The reference bankroll of 800 is the Kelly denominator, not a bankroll that is updated between hands. Bets do not shrink after a loss or grow after a win inside a shoe.

## What the numbers mean

Profit is reported per hand offered, including hands where the stake is only the minimum. Cluster-robust standard errors group hands dealt from the same shoe, and the intervals below are 95% normal approximations.

| Comparison | Hands | Result |
|---|---:|---|
| Untrained flat bet | 500,000 | −0.643 per hand |
| Q-learning, flat bet | 500,000 | −0.085 per hand |
| Infinite-deck play, flat bet | 2,000,000 | −0.00456 per hand |
| Composition network, flat bet | 2,000,000 | −0.00328 per hand |
| Network minus infinite-deck, flat bet | 2,000,000 | **+0.00128** (CI +0.00090 to +0.00165) |
| Both on one shared Hi-Lo schedule, every hand | 2,000,000 | network +0.02160, probability +0.01716, gap **+0.00445** |
| Network may bet 0 | 1,000,000 | +0.02784 per hand, average stake 1.38, bet on 27% of hands |
| Network bets at least 1 every hand | 1,000,000 | **+0.03049** (CI +0.02007 to +0.04092), average stake 2.59 |
| Full-Kelly infinite-deck play, bet at least 1, same shoes | 1,000,000 | +0.01606 (CI +0.00834 to +0.02379), max bet 12 |
| Always-bet network minus that full-Kelly baseline | 1,000,000 | **+0.01443** (CI +0.00963 to +0.01923) |

The gap against full-Kelly infinite-deck play is mostly the bet, not a new playing system. That baseline raises with a single true count and stops at 12, and it clips its edge estimate. This bot uses the full mix of remaining ranks and can bet up to 22.2, so more money sits on the shoes it likes. The playing advantage, measured with a flat bet, is about a tenth of a percent per hand.

An attempt to fine-tune play from actual hand outcomes, on top of imitation, made the paired holdout worse by 0.00677 per hand and was discarded.

Sitting out is still in `results/internal.txt` for the record. It is a different rule: 728,238 of those 1,000,000 hands were scored as a bet of zero while the shoe advanced anyway.

## Layout

```text
blackjack_solver/
  engine.py         shoe, Hi-Lo count, one hand
  exact.py          infinite-deck backward induction
  composition.py    the same induction at the current shoe mix
  network.py        NumPy MLP, Adam, cross-entropy
  internal.py       play, bet, and insurance networks; always-bet holdout
  same_holdout.py   flat-bet comparison against infinite-deck play
  count_holdout.py  both players on one shared Hi-Lo schedule
  betting.py        Kelly stake from a true-count regression
  stats.py          cluster-robust means, differences, and intervals
  experiment.py     the earlier Q-learning run
  qtable.py         tabular Q-learning and the insurance bins
  self_check.py     rules, exact values, and a small statistics check
viz/
  index.html        side-by-side replay
  record_replay.py  writes replay.json from a fresh training run
  replay.json       eight recorded hands
results/            holdout reports
```

## Running

```bash
python3 -m pip install -r requirements.txt
python3 -m blackjack_solver.self_check
```

The always-bet training and the 1,000,000-hand comparison:

```bash
python3 -m blackjack_solver.internal
```

That retrains the play network, samples shoe outcomes, fits the bet and insurance networks, and writes `results/always_bet.txt`. It takes a few minutes.

The animation replays eight hands from one shoe. The left table is an untrained network betting 1. The right table is the trained bot; on that stretch of the shoe its bets run from about 8 to 17.

```bash
PYTHONPATH=. python3 viz/record_replay.py
python3 -m http.server 8765 --directory viz
```

Then open `http://127.0.0.1:8765/`.

## Limits

This is a simulation of a specific rule set. It is not a claim about a particular casino’s cut card, penetration, or betting limits. The Kelly cap of 22.2 units on an 800-unit reference bankroll is large for a table that watches bet spreads. A short session still loses often: the per-hand standard deviation is several units, which is why the intervals above are from hundreds of thousands of hands, clustered by shoe.
