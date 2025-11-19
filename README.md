# Liar’s Deck – Bayesian Bluff Detection Engine

This repository contains a **card-only mathematical model** of the *Liar’s Deck* / *Liar’s Bar* minigame.  
It computes, **for each opponent claim**:

- how likely the claim is to be **card-feasible** (could be true, given the deck);  
- how likely a “typical human” is to **bluff** in such a spot;  
- the **posterior probability** that the opponent is bluffing vs telling the truth;  
- the **expected value (EV)** of calling “Liar!” given your risk preferences.

It also exposes a small **CLI tool** so you can drive the model by hand while playing.

---

## 1. Background and Sources

This model combines three components:

1. **Card combinatorics (hypergeometric distribution).**

   The deck is the classic Liar’s Deck used in *Liar’s Bar*:

   - 6 Kings, 6 Queens, 6 Aces, 2 Jokers → **20 cards total**;
   - The “table” is one of {Kings, Queens, Aces};
   - **Target cards** for a table are *all cards of that rank + all Jokers*.

   Given your hand and any revealed cards, we compute the probability that an
   opponent *could* hold enough target cards to make their claim truthful.

2. **Behavioural bluff priors** (human bluff tendencies).

   The bluff prior is inspired by empirical bluffing data from online poker,
   especially Palomäki et al. (2016), “To Bluff like a Man or Fold like a Girl?
   Gender Biased Deceptive Behaviour in Online Poker” (PLOS ONE).  
   Their data show that bluff frequency is not constant; it depends on context
   and perceived plausibility.

   In this basic version, we approximate that with a simple **piecewise prior**
   \(p_b(p_f)\) mapping feasibility → bluff rate.  
   In more advanced usage, you can fit a **logistic regression** from their
   `final_data_supplementary.sav` dataset and plug it in as a learned prior.

3. **Bayesian decision layer + EV rule.**

   We combine card-feasibility and bluff prior via a simple Bayesian-style
   calculation to produce

   \[
   P(\text{truth} \mid \text{claim}),\quad P(\text{bluff} \mid \text{claim}),
   \]

   then map that into an **expected value** for calling “Liar!” given asymmetric
   penalties (being wrong usually hurts more than being right helps).

This is meant to be a **transparent, interpretable baseline** that can be used:

- as a standalone “math coach” while playing; or
- as a belief module inside more complex RL agents (e.g. NFSP, CFR) for Liar’s Bar.

---

## 2. Installation and Requirements

The core file is:

- `liar.py` – everything (deck model, bluff model, EV logic, CLI) lives here.

### Python version

- Python **3.10+** recommended (for type hints like `list[int]` etc.).

### Dependencies

The main script uses **only the standard library**:

- `dataclasses`
- `math` (for `comb`)
- `typing` (for `Callable`, `Optional`)

No external packages are needed to run the CLI or use the basic API.

For advanced behavioural modelling (optional, not in this file) you may want:

- `pyreadstat` – to read `final_data_supplementary.sav` from Palomäki et al.
- `scikit-learn` – to fit a logistic regression `P(bluff | p_feasible)`.

---

## 3. Quickstart (CLI)

Run the script directly:

```bash
python liar.py
```

You’ll see:

```text
=== Liar's Deck: card-only model with bluff probabilities (CLI) ===
```

The CLI then asks:

1. **How many opponents?**  
   e.g. `3`.

2. **How many cards per player?**  
   e.g. `5`.

3. **Table type (K/Q/A)?**  
   The rank that is currently “live” on the table (Kings, Queens, or Aces).

4. **Your hand**:
   - how many cards you hold;
   - how many of those are target cards (table rank + Jokers).

5. **Revealed cards**:
   - number of target cards already revealed/out of play;
   - number of non-target cards already revealed/out of play.

After this, you enter a **turn loop**:

- Choose whose turn it is:
  - `"me"` – you play; the script updates your hand + revealed cards.
  - `"opponent"` – an opponent makes a claim; the script evaluates it.
  - `"q"` – quit.

### Opponent turn flow

For an opponent’s turn you’ll be asked:

1. Which opponent (1..N)?  
2. How many cards that opponent currently holds?  
3. How many **target cards** they **claim** to play this turn?

The script then prints something like:

```text
=== OPPONENT CLAIM EVALUATION ===
Opponent #1:
  Hand size before play           : 5
  Claimed target cards played     : 3
Card-feasible truth probability   : P(COULD be truthful) = 0.2418
Behavioural bluff prior           : P(bluff | state)     = 0.3000
Posterior P(truth | claim)        : 0.3607
Posterior P(bluff | claim)        : 0.6393
(Feasibility from cards + bluff priors from human data;
 still only an estimate.)
EV(call) = -0.0820  → Recommendation: DO NOT CALL
```

Interpretation:

- **P(COULD be truthful) = 0.2418**  
  Given the deck, your hand, and revealed cards, there is a 24.18% chance
  that this opponent could actually hold ≥3 target cards in their 5-card hand.

- **P(bluff | state) = 0.3000**  
  Behavioural prior: humans bluff 30% of the time in situations with this level
  of implausibility.

- **Posterior P(truth | claim) = 0.3607**  
  After combining the two stories (truthful-but-feasible vs bluff), we estimate
  a 36.07% chance the claim is true and 63.93% chance it’s a bluff.

- **EV(call) = -0.0820**  
  Under the configured risk parameters (`gain_if_correct=1`, `loss_if_wrong=2`),
  the expected value of calling “Liar!” is slightly negative, so the bot
  recommends **not calling**.

You can then optionally tell the script how many cards from this play were
actually revealed; the deck and hand counts are updated for the next turn.

---

## 4. Programmatic API

You can also import and use the model from other Python code:

```python
from liar import (
    DeckConfig,
    CardBeliefState,
    LiarDeckAgent,
    bluff_rate_from_feasibility,
    posterior_truth_probability,
    call_liar_ev,
    should_call_liar,
)

# 1. Deck and belief state
deck_cfg = DeckConfig()
state = CardBeliefState(
    deck_cfg=deck_cfg,
    table_rank="K",          # 'K', 'Q', or 'A'
    my_hand_size=5,
    my_target_cards=2,
    revealed_target_cards=1,
    revealed_non_target_cards=0,
)

# 2. Compute feasibility of an opponent's claim
p_feasible = state.prob_opponent_can_be_truthful(
    opp_hand_size=4,
    claimed_target_cards=3,
)

# 3. Behavioural bluff prior
p_bluff_prior = bluff_rate_from_feasibility(p_feasible)

# 4. Posterior truth probability
p_truth = posterior_truth_probability(p_feasible, p_bluff_prior)

# 5. EV and decision
ev_call = call_liar_ev(p_truth, gain_if_correct=1.0, loss_if_wrong=2.0)
should_call = should_call_liar(p_truth, gain_if_correct=1.0, loss_if_wrong=2.0)

print(p_feasible, p_bluff_prior, p_truth, ev_call, should_call)
```

Or use the **agent wrapper**:

```python
agent = LiarDeckAgent(
    state=state,
    gain_if_correct=1.0,
    loss_if_wrong=2.0,
)

opp_hand_size = 4
claimed_target_cards = 3

p_truth = agent.posterior_truth_for_claim(opp_hand_size, claimed_target_cards)
call = agent.decide_on_claim(opp_hand_size, claimed_target_cards)
```

---

## 5. Mathematics

### 5.1 Card-feasibility (hypergeometric distribution)

Let

- \(N_{\text{rem}}\) = number of cards still unknown to you;  
- \(T_{\text{rem}}\) = number of **target cards** among those unknown;  
- an opponent holds \(h\) cards and claims to play \(c\) target cards this turn.

Assume their hand is a random draw from the unknown cards:

\[
X \sim \mathrm{Hypergeometric}(N_{\text{rem}}, T_{\text{rem}}, h)
\]

where \(X\) is “number of target cards in opponent’s hand”.

The probability that their claim is **card-feasible** is:

\[
P(\text{feasible}) = P(X \ge c) =
\sum_{x=c}^{\min(T_{\text{rem}},h)}
\frac{\binom{T_{\text{rem}}}{x}\binom{N_{\text{rem}}-T_{\text{rem}}}{h-x}}
     {\binom{N_{\text{rem}}}{h}}.
\]

The code computes this via `_hypergeom_at_least`.

### 5.2 Behavioural bluff prior

Let

- \(p_f = P(\text{feasible})\) (the hypergeometric result).

We define a behavioural prior

\[
p_b = P(\text{bluff} \mid p_f),
\]

approximating how often a typical human bluffs in situations with feasibility
\(p_f\). In this simple version it is piecewise:

```python
if p_feasible < 0.10:
    p_bluff = 0.40
elif p_feasible < 0.30:
    p_bluff = 0.30
elif p_feasible < 0.60:
    p_bluff = 0.22
else:
    p_bluff = 0.10
```

Interpretation:

- if the claim is almost impossible, bluffing is relatively common (~40%);
- if the claim is very plausible, bluffing is rarer (~10%);
- in between, bluff rates interpolate via empirical heuristics.

A more advanced approach is to **learn** \(p_b(p_f)\) from the Palomäki dataset:

\[
\log\frac{p_b(p_f)}{1 - p_b(p_f)} = \beta_0 + \beta_1 p_f,
\]

fitted by logistic regression on observed bluffs vs non-bluffs.

### 5.3 Posterior truth / bluff probability

We use a simple two-hypothesis model:

- hypothesis T: “the player is truthful and the deck allows their claim”;
- hypothesis B: “the player is bluffing”.

We approximate:

\[
P(\text{truth} \cap \text{feasible}) = p_f \cdot (1 - p_b),
\]
\[
P(\text{bluff}) = p_b.
\]

The total weight is:

\[
p_{\text{tot}} = p_f(1 - p_b) + p_b.
\]

Then:

\[
P(\text{truth} \mid \text{claim}) =
\frac{p_f(1 - p_b)}{p_{\text{tot}}},
\qquad
P(\text{bluff} \mid \text{claim}) = 1 - P(\text{truth} \mid \text{claim}).
\]

This is implemented in `posterior_truth_probability`.

This model is deliberately simple and can be refined (e.g. by conditioning
bluffs on feasibility as well, or using a richer generative model).

### 5.4 Expected value of calling “Liar!”

Finally, we map the posterior into an **action recommendation**.

Let:

- \(p_T = P(\text{truth} \mid \text{claim})\),
- \(p_B = 1 - p_T\),
- `gain_if_correct` = payoff if you call and they were bluffing,
- `loss_if_wrong`  = positive magnitude of loss if you call and they were truthful.

The expected value of calling is:

\[
EV(\text{call}) = p_B \cdot \text{gain\_if\_correct}
                 - p_T \cdot \text{loss\_if\_wrong}.
\]

We recommend calling if \(EV(\text{call}) \ge 0\).

This is implemented by `call_liar_ev` and `should_call_liar`.

---

## 6. Extending the Behaviour Model (optional)

If you have access to the Palomäki dataset (`final_data_supplementary.sav`), you
can:

1. Compute a `p_feasible` value for each decision using this hypergeometric
   model (e.g. in a notebook).
2. Label each decision with `is_bluff` ∈ {0,1}.
3. Fit a logistic model:

   ```python
   from sklearn.linear_model import LogisticRegression

   X = df[["p_feasible"]].values
   y = df["is_bluff"].values.astype(int)

   model = LogisticRegression()
   model.fit(X, y)

   def learned_bluff_prior(p_feasible: float) -> float:
       return model.predict_proba([[p_feasible]])[0, 1]
   ```

4. Plug this into `LiarDeckAgent`:

   ```python
   agent = LiarDeckAgent(
       state=state,
       bluff_prior_fn=learned_bluff_prior,
       gain_if_correct=1.0,
       loss_if_wrong=2.0,
   )
   ```

This replaces the hand-tuned piecewise rule with a **data-driven** bluff prior.

---

## 7. Limitations and Assumptions

- The deck is assumed to be **perfectly shuffled**; opponent hands are treated
  as random conditional on what you know.
- Behaviour is modelled via a **single average bluff prior**, not
  player-specific styles.
- The Bayesian model is intentionally simple; it does not fully model the
  temporal game dynamics or multi-round signalling of Liar’s Bar.
- The EV rule is **myopic** (one-step) and does not consider future rounds.

Despite these simplifications, the engine is useful as:

- an **interpretable mathematical baseline** for bluff detection;
- a **feature generator** (feasibility, posterior, EV) for RL agents;
- a tool to build intuition about how deck structure and human bluff
  tendencies interact.

---

## 8. Citation

If you write about this model, you may want to cite:

- Palomäki, J., Yan, J., Modic, D., & Laakasuo, M. (2016).  
  *To Bluff like a Man or Fold like a Girl? Gender Biased Deceptive Behaviour
  in Online Poker.* PLOS ONE, 11(7), e0157838.

- (Optional) your own working paper / report describing how this engine is used
  in Liar’s Deck / Liar’s Bar experiments.
