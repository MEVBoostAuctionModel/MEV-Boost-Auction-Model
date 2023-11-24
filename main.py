import mesa
import pandas as pd
import numpy as np

from scipy.stats import poisson, norm
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from mesa import Agent, Model
from mesa.time import RandomActivation, SimultaneousActivation


class PlayerWithNaiveStrategy(Agent):
    def __init__(self, unique_id, model, pm, delay, probability):
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.private_signal_value = 0
        self.private_signal = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.probability = probability
        self.bid_count = 0

    def step(self):

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value

        if self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm

        self.bid_queue.append((self.bid, self.aggregated_signal))

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


class PlayerWithAdaptiveStrategy(Agent):
    def __init__(self, unique_id, model, pm, delay, probability):
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.private_signal = 0
        self.private_signal_value = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.probability = probability
        self.bid_count = 0

    def step(self):

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value
        # delta is a small constant value added to the current maximum bid
        delta = 0.0001

        if len(self.model.max_bids) >= 1:
            if self.aggregated_signal - self.pm > self.model.max_bids[-1] + delta:
                self.bid = self.model.max_bids[-1] + delta
            elif self.aggregated_signal - self.pm <= self.model.max_bids[-1] + delta and self.aggregated_signal > self.pm:
                self.bid = self.aggregated_signal - self.pm

        self.bid_queue.append((self.bid, self.aggregated_signal))

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


class PlayerWithLastMinute(Agent):
    def __init__(self, unique_id, model, pm, time_reveal_delta, time_estimate, delay, probability) :
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.time_reveal_delta = time_reveal_delta
        self.private_signal = 0
        self.private_signal_value = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.time_estimate = time_estimate
        self.probability = probability
        self.bid_count = 0

    def step(self):
        t = self.model.schedule.time

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value

        time_reveal = self.time_estimate - self.time_reveal_delta - self.individual_delay - self.model.global_delay

        if t < time_reveal:
            self.bid = 0
        elif t >= time_reveal and self.aggregated_signal > self.pm:
            self.bid = self.aggregated_signal - self.pm
        else :
            self.bid = 0

        self.bid_queue.append((self.bid, self.aggregated_signal))

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay:
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


class PlayerWithStealthStrategy(Agent):
    def __init__(self, unique_id, model, pm, time_reveal_delta, time_estimate, delay, probability):
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.time_reveal_delta = time_reveal_delta
        self.private_signal = 0
        self.private_signal_value = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.time_estimate = time_estimate
        self.probability = probability
        self.bid_count = 0

    def step(self):
        t = self.model.schedule.time  # get current time

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value

        time_reveal = self.time_estimate - self.time_reveal_delta - self.individual_delay - self.model.global_delay

        if t < time_reveal and self.model.public_signal_value > self.pm:
            self.bid = self.model.public_signal_value
        elif t >= time_reveal and self.aggregated_signal > self.pm:
            self.bid = self.aggregated_signal - self.pm
        else:
            self.bid = 0

        self.bid_queue.append((self.bid, self.aggregated_signal))

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay:
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


class PlayerWithBluffStrategy(Agent):
    def __init__(self, unique_id, model, pm, time_reveal_delta, time_estimate, bluff_value, delay, probability):
        super().__init__(unique_id, model)
        self.pm = pm
        self.bid = 0
        self.time_reveal_delta = time_reveal_delta
        self.bluff_value = bluff_value
        self.private_signal = 0
        self.private_signal_value = 0
        self.aggregated_signal = 0
        self.individual_delay = delay
        self.bid_queue = deque(maxlen=(self.individual_delay + self.model.global_delay))
        self.time_estimate = time_estimate
        self.probability = probability
        self.bid_count = 0

    def step(self) :
        t = self.model.schedule.time  # get current time

        # Aggregated signal
        self.aggregated_signal = self.model.public_signal_value + self.private_signal_value

        time_reveal = self.time_estimate - self.time_reveal_delta - self.individual_delay - self.model.global_delay

        if t < time_reveal :
            self.bid = self.bluff_value
        elif t >= time_reveal and self.aggregated_signal > self.pm :
            self.bid = self.aggregated_signal - self.pm
        else :
            self.bid = 0

        self.bid_queue.append((self.bid, self.aggregated_signal))

    def advance(self):
        if len(self.bid_queue) == self.individual_delay + self.model.global_delay :
            self.model.current_bids.append(self.bid_queue.popleft())
            self.model.bid_agents.append(self.unique_id)

        if len(self.bid_queue) == 1:
            self.bid_count += 1
        elif self.bid_queue[-1] != self.bid_queue[-2]:
            self.bid_count +=1


def get_current_bids(model):
    return model.show_current_bids


def get_bid_agents(model):
    return model.show_bid_agents


def get_public_signal(model):
    return model.public_signal_value


def get_private_signal_max(model):
    return model.private_signal_max


def get_aggregated_signal_max(model):
    return model.aggregated_signal_max


class Auction(Model):
    def __init__(self, N, A, L, S, B, rate_public_mean, rate_public_sd, rate_private_mean, rate_private_sd, T_mean, T_sd, delay) :
        self.num_naive = N
        self.num_adapt = A
        self.num_lastminute = L
        self.num_stealth = S
        self.num_bluff = B
        self.global_delay = delay

        # Initialize bids
        self.max_bids = []
        self.current_bids = []
        self.bid_agents = []
        self.winning_agents = []
        self.show_current_bids = []
        self.show_bid_agents = []

        self.schedule = SimultaneousActivation(self)

        # Initialize signals
        self.public_signal = 0
        self.public_signal_value = 0
        self.private_signal = 0
        self.private_signal_max = 0
        self.aggregated_signal_max = 0
        self.probability_mean = 0
        self.probability_std = 0

        # Initialize winner's info
        self.winner_profit = 0
        self.winner_aggregated_signal = 0
        self.winner_probability = 0
        self.auction_efficiency = 0


        # Initialize auction time and rate parameters
        self.T = norm.rvs(loc=T_mean, scale=T_sd)

        self.public_lambda = norm.rvs(loc=rate_public_mean, scale=rate_public_sd)

        self.private_lambda = norm.rvs(loc=rate_private_mean, scale=rate_private_sd)

        probabilities = [i * 0.01 + 0.8 for i in range(15)]

        # Create Agents
        for i in range(self.num_naive):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            delay = 1
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithNaiveStrategy(i, self, pm, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_adapt):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            delay = 1
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithAdaptiveStrategy(i + self.num_naive, self, pm, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_lastminute):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            time_reveal_delta = random.randint (10,20)
            time_estimate = 1200
            delay = 1
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithLastMinute(i + self.num_naive + self.num_adapt, self, pm,
                                     time_reveal_delta, time_estimate, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_stealth):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            time_reveal_delta = random.randint (10,20)
            time_estimate = 1200
            delay = 1
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithStealthStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute, self, pm,
                                         time_reveal_delta, time_estimate, delay, probability)
            self.schedule.add(a)

        for i in range(self.num_bluff):
            pm = norm.rvs(loc=0.00659, scale=0.0001)
            time_reveal_delta = random.randint (10,20)
            time_estimate = 1200
            bluff_value = np.random.uniform(0.25, 0.27)
            delay = 1
            probability = np.random.uniform(0.8, 1.0)
            a = PlayerWithBluffStrategy(i + self.num_naive + self.num_adapt + self.num_lastminute + self.num_stealth,
                                        self, pm, time_reveal_delta, time_estimate, bluff_value, delay, probability)
            self.schedule.add(a)

        # Initialize data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Current Bids": get_current_bids,
                "Agents": get_bid_agents,
                "Public Signal": get_public_signal,
                "Private Signal Max": get_private_signal_max,
                "Aggregated Signal Max": get_aggregated_signal_max
            },
            agent_reporters={"Bid": "bid",
                             "Probability": "probability",
                             "Bid Count": "bid_count"
                             }
        )

    def step(self):

        # Update public signal
        new_public_signal = poisson.rvs(mu=self.public_lambda)
        self.public_signal += new_public_signal

        for _ in range(new_public_signal) :
            signal_value = np.random.lognormal(mean=-11.66306, sigma=3.05450)
            # Add the value of the current public signal to the total value
            self.public_signal_value += signal_value

        # Update private signal
        new_private_signal = poisson.rvs(mu=self.private_lambda)
        self.private_signal += new_private_signal

        for _ in range (new_private_signal):
            private_signal_value = np.random.lognormal(mean=-8.41975, sigma =1.95231)
            self.private_signal_max += private_signal_value
            for agent in self.schedule.agents:
                if random.random() < agent.probability:
                    agent.private_signal_value += private_signal_value

        self.aggregated_signal_max = self.public_signal_value + self.private_signal_max

        self.schedule.step()

        # Select the winner of the step
        if self.current_bids:
            max_bid, aggregated_signal_at_time_of_bid = max(self.current_bids, key=lambda x: x[0])
            self.max_bids.append(max_bid)
            winner_id = self.bid_agents[self.current_bids.index((max_bid, aggregated_signal_at_time_of_bid))]
            self.winning_agents.append(winner_id)

            if self.aggregated_signal_max == 0:
                self.auction_efficiency = 0
            else:
                self.auction_efficiency = max_bid/self.aggregated_signal_max

            for agent in self.schedule.agents:
                if agent.unique_id == winner_id:
                    self.winner_profit = aggregated_signal_at_time_of_bid - max_bid
                    self.winner_aggregated_signal = aggregated_signal_at_time_of_bid
                    self.winner_probability = agent.probability




        self.show_current_bids = self.current_bids.copy()
        self.show_bid_agents = self.bid_agents.copy()

        self.current_bids.clear()
        self.bid_agents.clear()

        # Collect data at the end of the step
        self.datacollector.collect(self)


# Setup and run the model
model = Auction(2, 2, 2, 2, 2, rate_public_mean=0.085, rate_public_sd=0, rate_private_mean=0.04, rate_private_sd=0,
                T_mean=12, T_sd=0, delay=1)

for i in range(int(model.T * 100)):
    model.step()

# Data Collection
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()

# Print the simulation details
print(f"Public signal number: {model.public_signal}")
print(f"Public signal value: {model.public_signal_value}")
print(f"Private signal number: {model.private_signal}")
print(f"Auction time (seconds): {model.T}")
print(f"Auction time steps: {model.schedule.time}")
print("\n")
print(agent_data.loc[i, 'Probability'])
print("\n")
print(agent_data.loc[i, 'Bid Count'])
print("\n")
print(f"Winning Bid of Each Step: {model.max_bids}")
print(f"Winning Agent of Each Step: {model.winning_agents}")


# Print table for each step
for time_step in range(int(model.T * 100)):
    current_bids_at_time_step = model_data.loc[time_step, 'Current Bids']
    bid_agents_at_time_step = model_data.loc[time_step, 'Agents']
    if current_bids_at_time_step :
        bids, aggregated_signals = zip(*current_bids_at_time_step)

        df = pd.DataFrame({
            'Bids' : bids,
            'Aggregated Signals' : aggregated_signals,
            'Agents' : bid_agents_at_time_step
        })
        print(f"Data for time step {time_step}:")
        print(df)
        print("\n")
    else :
        print(f"Data for time step {time_step}: Empty Dataframe. No bids recorded on the relay.\n")

print('Winning Agent ID: ' + str(model.winning_agents[-1:][0]))
print('Winning bid value: ' + str(model.max_bids[-1:][0]))
print(f"Winner Profit: {model.winner_profit}")
print(f"Winner Total Signal: {model.winner_aggregated_signal}")
print(f"Winner Probility: {model.winner_probability}")
print('Winning bid time: ' + str(time_step) + ' ms')
print(f"Auction Efficiency: {model.auction_efficiency}")
print(model.aggregated_signal_max-model.winner_aggregated_signal)

dfs = []

for i in range(len(model_data)) :
    current_bids = model_data["Current Bids"].iloc[i]
    current_agents = model_data["Agents"].iloc[i]

    bids, aggregated_signals = zip(*current_bids) if current_bids else ([], [])

    bid_dict = {}
    for agent, bid, signal in zip(current_agents, bids, aggregated_signals) :
        bid_dict[f"Agent_{agent}_Bid"] = bid
        bid_dict[f"Agent_{agent}_Signal"] = signal

    # Convert the dictionary to a DataFrame and store it in the list
    dfs.append(pd.DataFrame(bid_dict, index=[i]))

all_bids = pd.concat(dfs)

public_signals = model_data["Public Signal"]
private_signal_max = model_data["Private Signal Max"]
aggregated_signal_max = model_data["Aggregated Signal Max"]


plt.figure(figsize=(20, 12))
plt.gca().set_prop_cycle('color', plt.cm.inferno(np.linspace(0, 1, len(all_bids.columns))))

fontsize = 12
for column in all_bids.columns:
    plt.plot(all_bids.index, all_bids[column], label=column,linewidth=2)
plt.plot(public_signals.index, public_signals, label='Public Signal', linewidth=3.5, color='green')
plt.plot(private_signal_max.index, private_signal_max, label='Private Max', linewidth=3.5, color='blue')
plt.plot(aggregated_signal_max.index, aggregated_signal_max, label='Aggregated Max', linewidth=3.5, color='indigo')
plt.xlabel('Time Step', fontsize=fontsize)
plt.ylabel('Bid Value', fontsize=fontsize)
plt.legend(title='Agent ID', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.title('Bids Received by Relay Across All Time Steps', fontsize=fontsize)
plt.grid(False)
plt.xticks(np.arange(0, 1300, 100), fontsize=fontsize)
plt.yticks(np.arange(0, 0.27, 0.01), fontsize=fontsize)
plt.axvline(1200, color='k')
plt.xlim(0), plt.ylim(0)
plt.show()


