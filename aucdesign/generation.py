from aucdesign.ad_auction import AdSpot, Bidder, Platform
import os
import random
from scipy.stats import lognorm, bernoulli
from math import exp
import pandas as pd
from tqdm import tqdm


def inequality(data):
    """Determine the inequality of a collection of auctions

    Parameters
    -----------
        data: collected information about previous auction outcomes

    Returns
    -----------
        inequality_score: measure of inequality in range [-1, 1]
    """
    probability_pos = sum(data["+"]) / len(data["+"]) if len(data["+"]) != 0 else 0
    probability_neg = sum(data["-"]) / len(data["-"]) if len(data["-"]) != 0 else 0

    inequality_score = probability_pos - probability_neg
    return inequality_score


def fairness(inequality_score, inequality_steps):
    """Fairness factor used to adjust ad allocation

    Parameters
    -----------
        inequality_score: calculated inequality of previous auctions
        group: group for which bid is being considered (either + or -)
        inequality_steps: number of auctions since an equal auction has taken place

    Returns
    -----------
        fairness_factor: the final factor that can be incorporated into bids to ensure fairness
    """
    # Calculate PI signal
    signal = 5 * inequality_score * inequality_steps + 3 * inequality_score
    # Capping signal to avoid OverflowError from exp(.)
    signal = abs(signal) if abs(signal) <= 15 else 15
    # Transform signal using logistic function
    signal = 1 / (1 + exp(-signal))

    fairness_factor = 1 - signal
    return fairness_factor


def simple_valuation(bidder, adspot, ctrs):
    """Compute the bidder's valuation for the adspot by summing the values for each tag

    Notice that missing tags contribute zero.

    Parameters
    -----------
        bidder: Bidder instance
        adspot: AdSpot instance
        ctrs: List of click-through rates (not used in this simple function)

    Returns
    -----------
        val: The computed valuation for this adspot.
    """
    val = 0.0
    for t in adspot.tags:
        val += bidder.targeting.get(t, 0.0)
    return val


def run_generation() -> None:
    n_auctions = 25  # Total number of auction (both + and -)
    n_bidders = 15
    n_trajectories = 500

    random_seed = 42


    results = {}

    # Independent auction trajectories with different seeds
    for trajectory in tqdm(range(n_trajectories * 2)):
        if trajectory < n_trajectories:
            prob_plus = 0.4
            prob_minus = 0.6
        else:
            prob_plus = 0.7
            prob_minus = 0.3

        random_seed += n_auctions
        inequality_steps = 0  # Number of auctions since inequality 0
        inequality_history = []  # Inequality after each auction in trajectory
        inequality_history_ctrl = []  # Inequality without correcting measures

        # Tracking auction results within trajectory,
        # True if a high-stakes ad won
        collected_data = {"+": [], "-": []}
        collected_data_ctrl = {"+": [], "-": []}

        # Single trajectory consisting of repeated auctions
        for _ in range(n_auctions):
            random.seed(random_seed)

            # Select group that is targeted in current loop iteration auction
            auction_tag = ["+"] if random.choice([0, 1]) else ["-"]
            auction = AdSpot(num_slots=1, tags=auction_tag, pos=[0.9])

            # Generate bidders and associated bids
            bidders = []
            bidders_ctrl = []
            fairness_score = fairness(inequality(collected_data), inequality_steps)
            for j in range(n_bidders):
                target = {}
                target_ctrl = {}
                target_plus = float(lognorm.rvs(1, 4, 1, random_state=random_seed))
                target_minus = float(lognorm.rvs(1, 2.5, 2, random_state=random_seed))

                if auction_tag == ["+"] and bernoulli.rvs(
                    prob_plus, random_state=random_seed
                ):
                    target["+"] = target_plus * fairness_score
                    target_ctrl["+"] = target_plus
                if auction_tag == ["-"] and bernoulli.rvs(
                    prob_minus, random_state=random_seed
                ):
                    target["-"] = target_minus * fairness_score
                    target_ctrl["-"] = target_minus
                if len(target.keys()) == 0:
                    target["+"] = target_plus
                    target_ctrl["+"] = target_plus
                    target["-"] = target_minus
                    target_ctrl["-"] = target_minus

                bidders.append(Bidder(str(j), target))
                bidders_ctrl.append(Bidder(str(j), target_ctrl))
                random_seed += 1

            # Calculate outcome of auction
            platform = Platform(bidders)
            platform_ctrl = Platform(bidders_ctrl)
            outcome = platform.assign(
                [auction], method="first_price", valuation_fn=simple_valuation
            )[0]
            outcome_ctrl = platform_ctrl.assign(
                [auction], method="first_price", valuation_fn=simple_valuation
            )[0]

            # Store outcome of auction
            high_stakes_winner = len(outcome["winners"][0].targeting.keys()) == 2
            high_stakes_winner_ctrl = (
                len(outcome_ctrl["winners"][0].targeting.keys()) == 2
            )
            if auction_tag == ["+"]:
                collected_data["+"].append(high_stakes_winner)
                collected_data_ctrl["+"].append(high_stakes_winner_ctrl)
            elif auction_tag == ["-"]:
                collected_data["-"].append(high_stakes_winner)
                collected_data_ctrl["-"].append(high_stakes_winner_ctrl)

            inequality_steps = (
                inequality_steps + 1 if inequality(collected_data) != 0 else 0
            )
            inequality_history.append(inequality(collected_data))
            inequality_history_ctrl.append(inequality(collected_data_ctrl))

        # Save inequality_history to results
        if trajectory < n_trajectories:
            results["type1_" + str(trajectory)] = inequality_history
            results["type1_ctrl_" + str(trajectory)] = inequality_history_ctrl
        else:
            results["type2_" + str(trajectory)] = inequality_history
            results["type2_ctrl_" + str(trajectory)] = inequality_history_ctrl

    # Save the collected results as a CSV
    results = pd.DataFrame(results)
    os.makedirs("results/", exist_ok=True)
    results.to_csv("results/data.csv", index=False)

