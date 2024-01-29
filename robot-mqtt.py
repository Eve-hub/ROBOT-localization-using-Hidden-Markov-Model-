import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import re

# Read transition matrix from CSV
csv_file_path_transition = 'Matrix.csv'
df_transition = pd.read_csv(csv_file_path_transition, sep=';', header=None)
transition = df_transition.to_numpy()

# Read emission matrix from CSV
csv_file_path_emission = 'emission.csv'
df_emission = pd.read_csv(csv_file_path_emission, sep=';', header=None)
emission = df_emission.to_numpy()

# Define states and symbols
states = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
          'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31',
          'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46',
          'S47', 'S48', 'S49']

states_dic = {f'S{i + 1}': i for i in range(len(states))}
sequence_syms = {'N': 0, 'NS': 1, 'NW': 2, 'NE': 3, 'E': 4, 'ES': 5, 'EW': 6, 'W': 7, 'WS': 8, 'S': 9, 'NOdet': 10}

# Test sequence
test_sequence = ['W', 'NS', 'NS', 'NOdet', 'NS', 'NS', 'E']
print(test_sequence)

# Node values stored during viterbi forward algorithm
node_values = np.zeros((len(states), len(test_sequence)))

# Probabilities of going to end state
end_probs = [0.1] * len(states)

# Probabilities of going from start state
start_probs = [0.5] * len(states)

# Storing max symbol for each stage
max_syms = [['' for _ in range(len(test_sequence))] for _ in range(len(states))]

# MQTT broker settings
broker_address = "192.168.43.130"  # Replace with your MQTT broker address
broker_port = 1883  # Replace with your MQTT broker port
topic = "robot/test"  # Change the topic name to "test"

# Create an MQTT client instance
client = mqtt.Client()

# Connect to the MQTT broker
client.connect(broker_address, broker_port, 60)

# Run a loop to publish the max_likely_states every 2 seconds
while True:
    for i, sequence_val in enumerate(test_sequence):
       for j in range(len(states)):
          # If the first sequence value, then do this
           if i == 0:
              node_values[j, i] = start_probs[j] * emission[j, sequence_syms[sequence_val]]
         # Else perform this
           else:
              values = [node_values[k, i - 1] * emission[j, sequence_syms[sequence_val]] * transition[k, j] for k in
                      range(len(states))]

              max_idx = np.argmax(values)
              max_val = max(values)
              max_syms[j][i] = states[max_idx]
              node_values[j, i] = max_val

    # End state value
    end_state = np.multiply(node_values[:, -1], end_probs)
    end_state_val = max(end_state)
    end_state_max_idx = np.argmax(end_state)
    end_state_sym = states[end_state_max_idx]

    # Obtaining the maximum likely states
    max_likely_states = [end_state_sym]

    prev_max = end_state_sym
    for count in range(1, len(test_sequence)):
       current_state = max_syms[states_dic[prev_max]][-count]
       max_likely_states.append(current_state)
       prev_max = current_state

    max_likely_states = max_likely_states[::-1]
    print(max_likely_states)
    
    # Convert state names to numbers using regular expression
    max_likely_states = [int(re.search(r'\d+', state).group()) for state in max_likely_states]

    print(max_likely_states)


    # Convert the max_likely_states list to a string for publishing
    max_likely_states_str = ",".join(map(str, max_likely_states))


    # Publish the max_likely_states to the MQTT topic
    client.publish(topic, max_likely_states_str)

    # Wait for 2 seconds before publishing again
    time.sleep(15)
    
# Disconnect from the MQTT broker (Note: This line will not be reached in this example)
client.disconnect()
