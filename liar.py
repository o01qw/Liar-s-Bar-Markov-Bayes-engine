"""
Liar's Deck / Liar's Bar – card-only model with turn-by-turn updates,
augmented with behavioural bluff probabilities.

Features:
- Classic Liar's Deck: 6 Kings, 6 Queens, 6 Aces, 2 Jokers (20 cards total).
- For a given table (Kings / Queens / Aces), "target" cards are:
    all cards of that rank + all Jokers (wild).
- You can:
    * set number of opponents and cards per player,
    * set the table type (Kings / Queens / Aces),
    * specify your hand size and how many target cards you hold,
    * track how many target and non-target cards have been revealed,
    * run a loop of turns where EITHER:
        - an opponent acts: we compute
              (a) P(the opponent COULD be truthful) from card math, and
              (b) P(truth | claim), P(bluff | claim) using bluff priors
          then you enter what was actually revealed, and we update deck + hands;
        - you act: you enter how many cards you played, how many were targets,
          and how many are now revealed, and we update your hand + deck.

Card side:
- P(opponent COULD be truthful given the cards) = P(X >= claim)
  where X ~ Hypergeometric(N_rem, T_rem, opp_hand_size).

Behaviour side:
- We map that feasibility into an estimated bluff rate using priors
  inspired by human bluff data (e.g. Palomäki et al. style results),
  then combine them in a simple Bayes-style way to get:
    P(truth | claim) and P(bluff | claim).

This still does NOT know what your actual opponent is thinking; it just
combines objective card feasibility with population-level bluff tendencies.
"""

from dataclasses import dataclass
from math import comb
from typing import Callable, Optional


# ---------- Deck configuration ----------

@dataclass
class DeckConfig:
    num_kings: int = 6
    num_queens: int = 6
    num_aces: int = 6
    num_jokers: int = 2

    @property
    def total_cards(self) -> int:
        return self.num_kings + self.num_queens + self.num_aces + self.num_jokers

    def total_target_cards(self, table_rank: str) -> int:
        """
        Number of target cards for the given table:
        rank + all jokers.
        table_rank: 'K', 'Q', or 'A'
        """
        table_rank = table_rank.upper()
        if table_rank == "K":
            base = self.num_kings
        elif table_rank == "Q":
            base = self.num_queens
        elif table_rank == "A":
            base = self.num_aces
        else:
            raise ValueError("table_rank must be 'K', 'Q', or 'A'")
        return base + self.num_jokers


# ---------- Belief state about the deck ----------

@dataclass
class CardBeliefState:
    """
    Belief state about the deck from your perspective.

    deck_cfg               : deck configuration
    table_rank             : 'K' / 'Q' / 'A'
    my_hand_size           : how many cards you currently hold
    my_target_cards        : how many of your hand cards are target cards
    revealed_target_cards  : how many target cards have been revealed / seen
    revealed_non_target_cards : how many non-target cards have been revealed / seen
    """
    deck_cfg: DeckConfig
    table_rank: str
    my_hand_size: int
    my_target_cards: int
    revealed_target_cards: int = 0
    revealed_non_target_cards: int = 0

    def remaining_totals(self) -> tuple[int, int]:
        """
        Returns:
            N_rem — how many cards remain unknown to you,
            T_rem — how many of those remaining cards are target cards.
        """
        total_cards = self.deck_cfg.total_cards
        total_target = self.deck_cfg.total_target_cards(self.table_rank)

        known_cards = (
            self.my_hand_size
            + self.revealed_target_cards
            + self.revealed_non_target_cards
        )

        if known_cards > total_cards:
            raise ValueError("Known cards exceed deck size.")
        if self.my_target_cards > self.my_hand_size:
            raise ValueError("my_target_cards cannot exceed my_hand_size.")
        if self.revealed_target_cards > total_target:
            raise ValueError("revealed_target_cards exceed total target cards in deck.")

        N_rem = total_cards - known_cards

        target_gone = self.my_target_cards + self.revealed_target_cards
        T_rem = total_target - target_gone

        if T_rem < 0:
            raise ValueError("Remaining target card count went negative.")
        return N_rem, T_rem

    @staticmethod
    def _hypergeom_at_least(k_min: int, N: int, K: int, n: int) -> float:
        """
        P(X >= k_min) for X ~ Hypergeometric(N, K, n).

        N    : population size
        K    : number of "success" states in population (target cards)
        n    : number of draws (hand size)
        k_min: minimum number of successes
        """
        if n > N:
            return 0.0
        max_success = min(K, n)
        if k_min > max_success:
            return 0.0
        denom = comb(N, n)
        if denom == 0:
            return 0.0
        num = 0
        for x in range(k_min, max_success + 1):
            num += comb(K, x) * comb(N - K, n - x)
        return num / denom

    def prob_opponent_can_be_truthful(
        self, opp_hand_size: int, claimed_target_cards: int
    ) -> float:
        """
        P(opponent COULD be truthful based on cards alone):

        Let:
            N_rem — remaining unknown cards,
            T_rem — target cards among them.
        Then:
            X ~ Hypergeometric(N_rem, T_rem, opp_hand_size)
            P(can be truthful) = P(X >= claimed_target_cards).
        """
        if opp_hand_size < 0 or claimed_target_cards < 0:
            raise ValueError("Hand size and claimed cards must be >= 0.")

        N_rem, T_rem = self.remaining_totals()

        if opp_hand_size > N_rem:
            raise ValueError("Opponent's hand size exceeds remaining unknown cards.")

        return self._hypergeom_at_least(
            k_min=claimed_target_cards,
            N=N_rem,
            K=T_rem,
            n=opp_hand_size,
        )


# ---------- Behavioural bluff model ----------

def bluff_rate_from_feasibility(p_feasible: float) -> float:
    """
    Map card-feasibility to an estimated bluff rate.

    Idea (inspired by human bluff data such as Palomäki-style experiments):
      - When a claim is almost impossible, honest play is rare; bluffs are common.
      - When a claim is moderately feasible, bluffing is less common.
      - When a claim is very feasible, we expect relatively few bluffs.

    You can tweak the numeric values to fit actual statistics you extract
    from the dataset.

    p_feasible: P(claim COULD be truthful) from pure card math.
    returns:    estimated P(bluff | this state) for an average opponent.
    """
    if p_feasible < 0.10:        # essentially "impossible" region
        return 0.40              # ~40% of such claims are bluffs
    elif p_feasible < 0.30:      # very unlikely but not impossible
        return 0.30
    elif p_feasible < 0.60:      # medium feasibility
        return 0.22
    else:                        # highly feasible region
        return 0.10              # ~10% bluff rate when truth is easy


def posterior_truth_probability(p_feasible: float, p_bluff: float) -> float:
    """
    Combine card feasibility and bluff prior in a simple Bayes-style way.

    - p_feasible: P(claim could be true given the deck).
    - p_bluff:    assumed P(bluff | this situation) from behavioural data.

    Model:
      P(truth & feasible) = p_feasible * (1 - p_bluff)
      P(bluff)            = p_bluff
      P(truth | claim)    = P(truth & feasible) / (P(truth & feasible) + P(bluff))

    This is a simple heuristic; you can refine it later.
    """
    p_truth_and_feasible = p_feasible * (1.0 - p_bluff)
    p_total = p_truth_and_feasible + p_bluff
    if p_total == 0:
        return 0.0
    return p_truth_and_feasible / p_total


# ---------- Decision layer: call/fold based on EV ----------

def call_liar_ev(
    p_truth: float,
    gain_if_correct: float = 1.0,
    loss_if_wrong: float = 1.0,
) -> float:
    """
    Expected value of calling "Liar!".

    Parameters
    ----------
    p_truth : float
        Posterior probability that the claim is actually truthful.
    gain_if_correct : float
        Payoff (utility) if you call and the opponent was bluffing.
    loss_if_wrong : float
        Positive payoff magnitude if you call and the opponent was truthful;
        this will be treated as a loss.

    Returns
    -------
    float
        Expected value of calling. Positive => calling is better than folding
        under this simple one-step model.
    """
    p_bluff = 1.0 - p_truth
    return p_bluff * gain_if_correct - p_truth * loss_if_wrong


def should_call_liar(
    p_truth: float,
    gain_if_correct: float = 1.0,
    loss_if_wrong: float = 1.0,
) -> bool:
    """
    Decide whether to call "Liar!" from expected value.

    Returns True iff EV(call) >= 0.
    """
    return call_liar_ev(p_truth, gain_if_correct, loss_if_wrong) >= 0.0


# ---------- Agent wrapper (top-level, not nested) ----------

@dataclass
class LiarDeckAgent:
    """
    Thin wrapper around CardBeliefState + bluff prior + EV rule.

    You can use this inside a bot loop or plug it into an RL environment
    as a model-based opponent.
    """
    state: CardBeliefState
    bluff_prior_fn: Callable[[float], float] = bluff_rate_from_feasibility
    gain_if_correct: float = 1.0
    loss_if_wrong: float = 1.0

    def posterior_truth_for_claim(
        self,
        opp_hand_size: int,
        claimed_target_cards: int,
    ) -> float:
        """
        Compute P(truth | claim, state) combining:

        - hypergeometric card math; and
        - behavioural bluff prior.
        """
        p_feasible = self.state.prob_opponent_can_be_truthful(
            opp_hand_size=opp_hand_size,
            claimed_target_cards=claimed_target_cards,
        )
        p_bluff = self.bluff_prior_fn(p_feasible)
        return posterior_truth_probability(p_feasible, p_bluff)

    def decide_on_claim(
        self,
        opp_hand_size: int,
        claimed_target_cards: int,
    ) -> bool:
        """
        High-level decision: should we call "Liar!"?

        Returns True if the call has non-negative EV under the
        (gain_if_correct, loss_if_wrong) parameters.
        """
        p_truth = self.posterior_truth_for_claim(
            opp_hand_size=opp_hand_size,
            claimed_target_cards=claimed_target_cards,
        )
        return should_call_liar(
            p_truth=p_truth,
            gain_if_correct=self.gain_if_correct,
            loss_if_wrong=self.loss_if_wrong,
        )


# ---------- Input helpers ----------

def ask_int(prompt: str, default: Optional[int] = None) -> int:
    while True:
        suffix = f" [default {default}]: " if default is not None else ": "
        s = input(prompt + suffix).strip()
        if not s and default is not None:
            return default
        try:
            return int(s)
        except ValueError:
            print("Please enter an integer.")


def ask_table_rank() -> str:
    s = input("Table type (K/Q/A or king/queen/ace): ").strip().lower()
    if s in ("k", "king", "kings"):
        return "K"
    if s in ("q", "queen", "queens"):
        return "Q"
    if s in ("a", "ace", "aces"):
        return "A"
    print("Unrecognised input, defaulting to Kings.")
    return "K"


def ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    while True:
        s = input(prompt + " [y/n]: ").strip().lower()
        if not s and default is not None:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def ask_actor() -> str:
    """
    Ask whose turn it is:
        'me'       – your own move,
        'opponent' – some opponent's move.
    """
    while True:
        s = input("Whose turn is it? (me/opponent, or 'q' to quit): ").strip().lower()
        if s in ("me", "m", "self", "you"):
            return "me"
        if s in ("opp", "opponent", "o"):
            return "opponent"
        if s == "q":
            return "quit"
        print("Please answer 'me', 'opponent' or 'q'.")


# ---------- Main turn loop ----------

def main():
    print("=== Liar's Deck: card-only model with bluff probabilities (CLI) ===\n")

    deck_cfg = DeckConfig()

    num_opponents = ask_int("How many opponents are at the table?", default=3)
    cards_per_player = ask_int("How many cards does each player start with?", default=5)

    # Track opponent hand sizes
    opp_hands = [cards_per_player] * num_opponents

    table_rank = ask_table_rank()
    table_name = {"K": "Kings", "Q": "Queens", "A": "Aces"}[table_rank]

    # Your hand
    my_hand_size = ask_int("How many cards do YOU currently hold?",
                           default=cards_per_player)
    my_target_cards = ask_int(
        f"How many of your cards are TARGET cards for this table ({table_name} + Jokers)? "
    )

    # Already revealed cards
    revealed_target_cards = ask_int(
        "How many TARGET cards have already been revealed / are known out of play?",
        default=0,
    )
    revealed_non_target_cards = ask_int(
        "How many NON-target cards have already been revealed / are known out of play?",
        default=0,
    )

    # Initialise belief state using the actual inputs
    state = CardBeliefState(
        deck_cfg=deck_cfg,
        table_rank=table_rank,
        my_hand_size=my_hand_size,
        my_target_cards=my_target_cards,
        revealed_target_cards=revealed_target_cards,
        revealed_non_target_cards=revealed_non_target_cards,
    )

    # Agent (for EV-based recommendations)
    agent = LiarDeckAgent(
        state=state,
        gain_if_correct=1.0,   # tweak these to reflect bullet risk / chips
        loss_if_wrong=2.0,
    )

    while True:
        # Show current deck state
        N_rem, T_rem = state.remaining_totals()
        print("\n--- CURRENT STATE ---")
        print(f"Table type              : {table_name}")
        print(f"Unknown cards remaining : N_rem = {N_rem}")
        print(f"Target cards remaining  : T_rem = {T_rem}")
        print(f"Your hand               : {state.my_hand_size} cards "
              f"(target: {state.my_target_cards})")
        print("Opponents' hand sizes   : " +
              ", ".join(f"#{i+1}={h}" for i, h in enumerate(opp_hands)))

        actor = ask_actor()
        if actor == "quit":
            print("Exiting turn loop.")
            break

        # -------- Your own turn --------
        if actor == "me":
            print("\n--- YOUR TURN ---")
            # Optionally correct your current hand size (if you know it's changed)
            state.my_hand_size = ask_int(
                "How many cards do you hold *before* this play?",
                default=state.my_hand_size,
            )

            played_total = ask_int(
                "How many cards do you play this turn (face-down or face-up)?",
                default=0,
            )
            if played_total < 0 or played_total > state.my_hand_size:
                print("Invalid number of cards played; skipping update.")
                continue

            played_target = ask_int(
                f"Of those {played_total}, how many are TARGET cards (for {table_name})?",
                default=min(played_total, state.my_target_cards),
            )

            # Update your hand content (regardless of whether they are revealed yet)
            state.my_hand_size -= played_total
            state.my_target_cards -= min(played_target, state.my_target_cards)

            if ask_yes_no("Did any of YOUR played cards become face-up / publicly known this turn?"):
                new_revealed_target = ask_int(
                    "How many of YOUR played cards became REVEALED TARGET cards?",
                    default=played_target,
                )
                new_revealed_non_target = ask_int(
                    "How many of YOUR played cards became REVEALED NON-target cards?",
                    default=max(0, played_total - new_revealed_target),
                )
                state.revealed_target_cards += new_revealed_target
                state.revealed_non_target_cards += new_revealed_non_target

                print(f"Updated: +{new_revealed_target} revealed target, "
                      f"+{new_revealed_non_target} revealed non-target from your play.")

            continue  # Move to next turn

        # -------- Opponent's turn --------
        if actor == "opponent":
            print("\n--- OPPONENT'S TURN ---")
            opp_index = ask_int(
                f"Which opponent is acting? (1..{num_opponents})",
                default=1,
            )
            if opp_index < 1 or opp_index > num_opponents:
                print("Invalid opponent index, using #1.")
                opp_index = 1
            idx = opp_index - 1

            opp_hand_size = ask_int(
                f"How many cards does opponent #{opp_index} hold BEFORE this play?",
                default=opp_hands[idx],
            )
            opp_hands[idx] = opp_hand_size

            claimed_target_cards = ask_int(
                f"How many TARGET cards does opponent #{opp_index} CLAIM to play now?",
                default=1,
            )

            # Compute feasibility and bluff-adjusted probabilities
            try:
                p_feasible = state.prob_opponent_can_be_truthful(
                    opp_hand_size=opp_hand_size,
                    claimed_target_cards=claimed_target_cards,
                )
            except ValueError as e:
                print(f"Error computing probability: {e}")
                continue

            p_bluff_prior = bluff_rate_from_feasibility(p_feasible)
            p_truth_given_claim = posterior_truth_probability(p_feasible, p_bluff_prior)
            p_bluff_given_claim = 1.0 - p_truth_given_claim

            print("\n=== OPPONENT CLAIM EVALUATION ===")
            print(f"Opponent #{opp_index}:")
            print(f"  Hand size before play           : {opp_hand_size}")
            print(f"  Claimed target cards played     : {claimed_target_cards}")
            print(f"Card-feasible truth probability   : P(COULD be truthful) = {p_feasible:.4f}")
            print(f"Behavioural bluff prior           : P(bluff | state)     = {p_bluff_prior:.4f}")
            print(f"Posterior P(truth | claim)        : {p_truth_given_claim:.4f}")
            print(f"Posterior P(bluff | claim)        : {p_bluff_given_claim:.4f}")
            print("(Feasibility from cards + bluff priors from human data; "
                  "still only an estimate.)")

            # EV-based recommendation
            ev_call = call_liar_ev(
                p_truth=p_truth_given_claim,
                gain_if_correct=agent.gain_if_correct,
                loss_if_wrong=agent.loss_if_wrong,
            )
            if ev_call >= 0:
                print(f"EV(call) = {ev_call:.4f}  → Recommendation: CALL LIAR")
            else:
                print(f"EV(call) = {ev_call:.4f}  → Recommendation: DO NOT CALL")

            # Update after reveal (if any cards became known)
            if ask_yes_no("Did any cards from this opponent's play become face-up / known?"):
                new_revealed_target = ask_int(
                    "How many REVEALED TARGET cards from this opponent's play?",
                    default=claimed_target_cards,
                )
                new_revealed_non_target = ask_int(
                    "How many REVEALED NON-target cards from this opponent's play?",
                    default=max(0, claimed_target_cards - new_revealed_target),
                )

                state.revealed_target_cards += new_revealed_target
                state.revealed_non_target_cards += new_revealed_non_target

                total_revealed_from_opp = ask_int(
                    "How many of these revealed cards definitely came from THIS opponent's hand?",
                    default=new_revealed_target + new_revealed_non_target,
                )
                opp_hands[idx] = max(0, opp_hands[idx] - total_revealed_from_opp)

                print(
                    f"Updated: +{new_revealed_target} revealed target, "
                    f"+{new_revealed_non_target} revealed non-target cards."
                )

            continue

    print("\nDone. Turn loop ended.")


if __name__ == "__main__":
    main()
