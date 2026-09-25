"""Six-deck blackjack shoe, Hi-Lo counter, and one-hand resolution."""

from __future__ import annotations

import numpy as np

from blackjack_solver.constants import (
    DOUBLE,
    HILO,
    HIT,
    N_DECKS,
    PENETRATION,
    SPLIT,
    STAND,
    STATE_DIM,
    SURRENDER,
)
from blackjack_solver.exact import hand_value

_HILO = np.asarray(HILO, dtype=np.int16)


class Hand:
    __slots__ = (
        "cards",
        "bet",
        "splits_left",
        "split_aces",
        "finished",
        "await_dealer",
        "profit",
        "own_tids",
        "ancestor_split_tids",
        "pending_hit",
    )

    def __init__(self, cards, bet, splits_left, split_aces=False, ancestors=None):
        self.cards = cards
        self.bet = float(bet)
        self.splits_left = int(splits_left)
        self.split_aces = bool(split_aces)
        self.finished = False
        self.await_dealer = False
        self.profit = None
        self.own_tids: list[int] = []
        self.ancestor_split_tids: list[int] = list(ancestors or [])
        self.pending_hit = None


class Shoe:
    def __init__(self, rng: np.random.Generator, n_decks: int = N_DECKS, penetration: float = PENETRATION):
        self.rng = rng
        self.n_decks = n_decks
        self.penetration = penetration
        self.total = 52 * n_decks
        self.cards = np.empty(self.total, np.int8)
        self.idx = 0
        self.cut = int(self.total * penetration)
        self.remaining = np.zeros(11, np.int16)
        self.running = 0
        self.hidden = None
        self.shuffle()

    def shuffle(self) -> None:
        i = 0
        for _deck_card in range(self.n_decks * 4):
            for rank in range(1, 10):
                self.cards[i] = rank
                i += 1
            for _ten in range(4):
                self.cards[i] = 10
                i += 1
        self.rng.shuffle(self.cards)
        self.idx = 0
        self.hidden = None
        self.remaining[:] = 0
        for rank in range(1, 10):
            self.remaining[rank] = 4 * self.n_decks
        self.remaining[10] = 16 * self.n_decks
        self.running = 0
        self.cut = int(self.total * self.penetration)

    def needs_shuffle(self) -> bool:
        return self.idx >= self.cut

    def draw(self) -> int:
        if self.idx >= self.total:
            raise RuntimeError("shoe exhausted")
        card = int(self.cards[self.idx])
        self.idx += 1
        return card

    def see(self, card: int) -> None:
        self.remaining[card] -= 1
        self.running += int(_HILO[card])

    def draw_seen(self) -> int:
        card = self.draw()
        self.see(card)
        return card

    def draw_hidden(self) -> int:
        card = self.draw()
        self.hidden = card
        return card

    def reveal_hidden(self) -> int:
        card = self.hidden
        if card is None:
            raise RuntimeError("no hidden card")
        self.see(card)
        self.hidden = None
        return card

    def unseen(self) -> int:
        return int(self.remaining[1:].sum())

    def true_count(self) -> float:
        unseen = self.unseen()
        if unseen <= 0:
            return 0.0
        return self.running / max(unseen / 52.0, 0.25)

    def fractions(self) -> np.ndarray:
        unseen = float(self.unseen())
        if unseen <= 0:
            return np.full(10, 0.1)
        return self.remaining[1:].astype(np.float64) / unseen

    def unseen_decks(self) -> float:
        return self.unseen() / 52.0

    def snapshot(self) -> tuple:
        return (
            self.cards[self.idx :].copy(),
            self.remaining.copy(),
            int(self.running),
        )

    @staticmethod
    def from_snapshot(snap: tuple, n_decks: int = N_DECKS) -> "Shoe":
        shoe = object.__new__(Shoe)
        suffix, remaining, running = snap
        shoe.rng = None
        shoe.n_decks = n_decks
        shoe.penetration = PENETRATION
        shoe.cards = suffix.copy()
        shoe.total = len(shoe.cards)
        shoe.idx = 0
        shoe.cut = shoe.total + 1
        shoe.remaining = remaining.copy()
        shoe.running = int(running)
        shoe.hidden = None
        return shoe


def encode_state(
    buf: np.ndarray,
    dealer_up: int,
    total: int,
    soft: int,
    pair_rank: int,
    can_double: bool,
    can_split: bool,
    can_surrender: bool,
    splits_left: int,
    true_count: float,
    fractions: np.ndarray,
    unseen_decks: float,
) -> np.ndarray:
    buf.fill(0.0)
    buf[dealer_up - 1] = 1.0
    if soft:
        if 12 <= total <= 21:
            buf[28 + (total - 12)] = 1.0
    elif 4 <= total <= 21:
        buf[10 + (total - 4)] = 1.0
    if pair_rank:
        buf[38 + (pair_rank - 1)] = 1.0
    buf[48] = 1.0 if can_double else 0.0
    buf[49] = 1.0 if can_split else 0.0
    buf[50] = 1.0 if can_surrender else 0.0
    buf[51] = float(np.clip(true_count, -10.0, 10.0) / 10.0)
    buf[52:62] = fractions
    buf[62] = unseen_decks / 6.0
    buf[63] = (splits_left / 3.0) if can_split else 0.0
    return buf


def settle(total: int, bet: float, dealer_total: int) -> float:
    if total > 21:
        return -bet
    if dealer_total > 21 or total > dealer_total:
        return bet
    if total < dealer_total:
        return -bet
    return 0.0


def _dealer_play(shoe: Shoe, up: int, hole: int) -> int:
    cards = [up, hole]
    total, _soft = hand_value(cards)
    while total < 17:
        cards.append(shoe.draw_seen())
        total, _soft = hand_value(cards)
    return total


def _is_blackjack(c1: int, c2: int) -> bool:
    return (c1 == 1 and c2 == 10) or (c2 == 1 and c1 == 10)


def _insurance_choice(decide_insurance, true_count: float, shoe: Shoe) -> bool:
    try:
        return bool(decide_insurance(true_count, shoe))
    except TypeError:
        return bool(decide_insurance(true_count))


def play_hand(shoe: Shoe, bet: float, decide, decide_insurance, on_decision=None) -> dict:
    """Play one round. `decide(state, legal, ctx) -> action`. Insurance sees only the true count."""
    empty = {
        "profit": 0.0,
        "play_profit": 0.0,
        "insurance_profit": 0.0,
        "posted_bet": 0.0,
        "transitions": [],
        "insurance_offered": False,
        "insurance_taken": False,
        "ten_density": None,
        "dealer_blackjack": False,
        "tc_ins": None,
        "player_blackjack": False,
    }
    if bet <= 0.0:
        return empty

    buf = np.empty(STATE_DIM, np.float64)
    c1 = shoe.draw_seen()
    c2 = shoe.draw_seen()
    up = shoe.draw_seen()
    ten_density = None
    tc_ins = None
    insurance_taken = False
    insurance_profit = 0.0
    insurance_offered = up == 1
    if insurance_offered:
        unseen = shoe.unseen()
        ten_density = float(shoe.remaining[10]) / unseen if unseen else 0.0
        tc_ins = shoe.true_count()
        insurance_taken = _insurance_choice(decide_insurance, tc_ins, shoe)

    hole = shoe.draw_hidden()
    dealer_bj = (up == 1 and hole == 10) or (up == 10 and hole == 1)
    player_bj = _is_blackjack(c1, c2)
    if insurance_offered and insurance_taken:
        insurance_profit = (1.0 if dealer_bj else -0.5) * bet

    if dealer_bj or player_bj:
        shoe.reveal_hidden()
        if dealer_bj and player_bj:
            play_profit = 0.0
        elif player_bj:
            play_profit = 1.5 * bet
        else:
            play_profit = -bet
        out = dict(empty)
        out.update(
            profit=play_profit + insurance_profit,
            play_profit=play_profit,
            insurance_profit=insurance_profit,
            posted_bet=bet,
            insurance_offered=insurance_offered,
            insurance_taken=insurance_taken,
            ten_density=ten_density,
            dealer_blackjack=dealer_bj,
            tc_ins=tc_ins,
            player_blackjack=player_bj,
        )
        return out

    transitions: list[dict] = []
    hands = [Hand([c1, c2], bet, 3)]
    i = 0
    while i < len(hands):
        hand = hands[i]
        while not hand.finished:
            total, soft = hand_value(hand.cards)
            if total >= 21:
                hand.finished = True
                hand.await_dealer = True
                hand.profit = None
                break
            pair_rank = 0
            if (
                len(hand.cards) == 2
                and hand.cards[0] == hand.cards[1]
                and not hand.split_aces
                and hand.splits_left > 0
            ):
                pair_rank = int(hand.cards[0])
            can_double = len(hand.cards) == 2 and not hand.split_aces
            can_split = pair_rank != 0
            can_surrender = len(hand.cards) == 2 and len(hands) == 1 and not hand.split_aces
            legal = np.zeros(5, np.float64)
            legal[HIT] = 1.0
            legal[STAND] = 1.0
            if can_double:
                legal[DOUBLE] = 1.0
            if can_split:
                legal[SPLIT] = 1.0
            if can_surrender:
                legal[SURRENDER] = 1.0
            fractions = shoe.fractions()
            tc = shoe.true_count()
            encode_state(
                buf,
                up,
                total,
                soft,
                pair_rank if can_split else 0,
                can_double,
                can_split,
                can_surrender,
                hand.splits_left,
                tc,
                fractions,
                shoe.unseen_decks(),
            )
            state = buf.copy()
            ctx = {
                "total": total,
                "soft": soft,
                "pair_rank": pair_rank,
                "up": up,
                "can_double": can_double,
                "can_split": can_split,
                "can_surrender": can_surrender,
                "splits_left": hand.splits_left,
                "tc": tc,
            }
            if hand.pending_hit is not None:
                transitions[hand.pending_hit]["next_ctx"] = ctx
                transitions[hand.pending_hit]["next_legal"] = legal.copy()
                hand.pending_hit = None
            action = int(decide(state, legal, ctx))
            if legal[action] <= 0.0:
                action = STAND
            if on_decision is not None:
                on_decision(state, legal, ctx, action)
            if action == STAND:
                tid = len(transitions)
                transitions.append({"state": state, "action": action, "ctx": ctx, "G": 0.0})
                hand.own_tids.append(tid)
                hand.finished = True
                hand.await_dealer = True
            elif action == SURRENDER:
                hand.profit = -0.5 * hand.bet
                tid = len(transitions)
                transitions.append({"state": state, "action": action, "ctx": ctx, "G": hand.profit})
                hand.own_tids.append(tid)
                hand.finished = True
            elif action == DOUBLE:
                hand.cards.append(shoe.draw_seen())
                hand.bet *= 2.0
                total, _soft = hand_value(hand.cards)
                tid = len(transitions)
                transitions.append({"state": state, "action": action, "ctx": ctx, "G": 0.0})
                hand.own_tids.append(tid)
                hand.finished = True
                if total > 21:
                    hand.profit = -hand.bet
                    hand.await_dealer = False
                else:
                    hand.await_dealer = True
            elif action == HIT:
                hand.cards.append(shoe.draw_seen())
                total, _soft = hand_value(hand.cards)
                tid = len(transitions)
                transitions.append({"state": state, "action": action, "ctx": ctx, "G": 0.0})
                hand.own_tids.append(tid)
                if total > 21:
                    hand.profit = -hand.bet
                    hand.finished = True
                    transitions[tid]["bust"] = True
                else:
                    hand.pending_hit = tid
            elif action == SPLIT:
                rank = int(hand.cards[0])
                tid = len(transitions)
                transitions.append({"state": state, "action": action, "ctx": ctx, "G": 0.0})
                ancestors = hand.ancestor_split_tids + [tid]
                left_card = shoe.draw_seen()
                right_card = shoe.draw_seen()
                child_splits = 0 if rank == 1 else hand.splits_left - 1
                left = Hand([rank, left_card], hand.bet, child_splits, rank == 1, ancestors)
                right = Hand([rank, right_card], hand.bet, child_splits, rank == 1, ancestors)
                if rank == 1:
                    left.finished = True
                    right.finished = True
                    left.await_dealer = True
                    right.await_dealer = True
                hands[i] = left
                hands.insert(i + 1, right)
                hand = hands[i]
            else:
                raise RuntimeError(f"bad action {action}")
        i += 1

    hole_card = shoe.reveal_hidden()
    if any(h.await_dealer for h in hands):
        dealer_total = _dealer_play(shoe, up, hole_card)
    else:
        dealer_total = 22
    for hand in hands:
        if hand.profit is None:
            total, _soft = hand_value(hand.cards)
            hand.profit = settle(total, hand.bet, dealer_total)
        elif hand.await_dealer:
            total, _soft = hand_value(hand.cards)
            hand.profit = settle(total, hand.bet, dealer_total)

    split_sums: dict[int, float] = {}
    for hand in hands:
        for tid in hand.own_tids:
            transitions[tid]["G"] = hand.profit / bet
        for tid in hand.ancestor_split_tids:
            split_sums[tid] = split_sums.get(tid, 0.0) + hand.profit / bet
    for tid, value in split_sums.items():
        transitions[tid]["G"] = value

    play_profit = float(sum(h.profit for h in hands))
    out = dict(empty)
    out.update(
        profit=play_profit + insurance_profit,
        play_profit=play_profit,
        insurance_profit=insurance_profit,
        posted_bet=bet,
        transitions=transitions,
        insurance_offered=insurance_offered,
        insurance_taken=insurance_taken,
        ten_density=ten_density,
        dealer_blackjack=False,
        tc_ins=tc_ins,
        player_blackjack=False,
    )
    return out
