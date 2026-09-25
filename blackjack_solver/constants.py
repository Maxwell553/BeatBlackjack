"""Shared constants for the blackjack rules and the network state layout."""

HIT, STAND, DOUBLE, SPLIT, SURRENDER = 0, 1, 2, 3, 4
N_ACTIONS = 5
ACTION_NAMES = ("HIT", "STAND", "DOUBLE", "SPLIT", "SURRENDER")

# Hi-Lo tags indexed by rank. Ace is 1, tens and faces are 10.
HILO = (0, -1, 1, 1, 1, 1, 1, 0, 0, 0, -1)

N_DECKS = 6
PENETRATION = 0.75

# Network input layout (length 64):
#  0:10   dealer upcard one-hot, ranks ace..ten
# 10:28   hard total one-hot, totals 4..21 (used when the hand is hard)
# 28:38   soft total one-hot, totals 12..21 (used when the hand is soft)
# 38:48   pair rank one-hot, ranks ace..ten (only if splitting is legal)
# 48:51   can double, can split, can surrender
# 51      true count / 10, clipped to [-1, 1]
# 52:62   unseen fraction of each rank ace..ten
# 62      unseen decks / 6
# 63      remaining splits / 3, if a split is legal
STATE_DIM = 64
