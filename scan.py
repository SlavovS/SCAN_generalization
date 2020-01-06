import keras
from keras.layers import Input, LSTM, TimeDistributed, Dense, SimpleRNN
from keras.models import Model
import keras.backend as K
import numpy as np

from keras.optimizers import Adam


actions = {"look":0, "run":1, "walk":2, "turn":3, "jump":4}

modifiers = {"right":0, "left":1, "twice":2, "thrice":3, "around":4, 
             "opposite":5, "and":6, "after":7}

outputs = {"i_turn_right":0, "i_turn_left":1, "i_look":2, "i_walk":3, 
           "i_run":4, "i_jump":5}


n_tokens = 6
n_actions = 5
n_modifiers = 8
n_outputs = 8 # the longest sequence of actions in the output (e.g. i_turn_left i_walk i_turn_left i_walk i_turn_left i_walk...)
n_output_actions = 6

def make_matrix_trial():
    
    trial_input_action = np.zeros((n_tokens, n_actions))
    trial_input_modifier = np.zeros((n_tokens, n_modifiers))

    trial_output_action = np.zeros((n_outputs, n_output_actions))
    
    tokens = np.random.permutation(n_tokens)
    

    
    jump = np.random.random()
    
    if jump > 0.9:
        trial_input_action[tokens[0], actions.get("jump")] = 1
        trial_output_action[0, outputs.get("i_jump")] = 1
        
    elif 0.7 <= jump <= 0.89:
        trial_input_action[tokens[0], actions.get("turn")] = 1
        trial_input_modifier[tokens[0], modifiers.get("right")] = 1
        trial_input_modifier[tokens[0], modifiers.get("twice")] = 1
        trial_output_action[0, outputs.get("i_turn_right")] = 1
        trial_output_action[1, outputs.get("i_turn_right")] = 1

    elif 0.55 <= jump <= 0.69:

        trial_input_action[tokens[0], actions.get("turn")] = 1
        trial_input_modifier[tokens[0], modifiers.get("left")] = 1
        trial_input_modifier[tokens[0], modifiers.get("thrice")] = 1
        trial_output_action[0, outputs.get("i_turn_left")] = 1
        trial_output_action[1, outputs.get("i_turn_left")] = 1
        trial_output_action[2, outputs.get("i_turn_left")] = 1
        
    elif 0.40 <= jump <= 0.54:
        
        trial_input_action[tokens[0], actions.get("look")] = 1
        trial_input_modifier[tokens[0], modifiers.get("right")] = 1
        trial_output_action[0, outputs.get("i_turn_right")] = 1
        trial_output_action[1, outputs.get("i_look")] = 1
        
    elif 0.25 <= jump <= 0.39:
        
        trial_input_action[tokens[0], actions.get("run")] = 1
        trial_input_modifier[tokens[0], modifiers.get("right")] = 1
        trial_input_modifier[tokens[0], modifiers.get("twice")] = 1
        trial_output_action[0, outputs.get("i_turn_right")] = 1
        trial_output_action[1, outputs.get("i_run")] = 1
        trial_output_action[2, outputs.get("i_turn_right")] = 1
        trial_output_action[3, outputs.get("i_run")] = 1
        
    elif 0.1 <= jump <= 0.24:
        trial_input_action[tokens[0], actions.get("walk")] = 1
        trial_input_modifier[tokens[0], modifiers.get("around")] = 1
        trial_input_modifier[tokens[0], modifiers.get("left")] = 1
        trial_input_modifier[tokens[0], modifiers.get("twice")] = 1
        trial_output_action[0, outputs.get("i_turn_left")] = 1
        trial_output_action[1, outputs.get("i_walk")] = 1
        trial_output_action[2, outputs.get("i_turn_left")] = 1
        trial_output_action[3, outputs.get("i_walk")] = 1
        trial_output_action[4, outputs.get("i_turn_left")] = 1
        trial_output_action[5, outputs.get("i_walk")] = 1
        trial_output_action[6, outputs.get("i_turn_left")] = 1
        trial_output_action[7, outputs.get("i_walk")] = 1
    
    else:
        
        trial_input_action[tokens[0], actions.get("walk")] = 1
        trial_input_modifier[tokens[0], modifiers.get("left")] = 1
        trial_output_action[0, outputs.get("i_turn_left")] = 1
        trial_output_action[1, outputs.get("i_walk")] = 1




    tokens = tokens[1:]

    return (
            [trial_input_action.reshape(1, n_tokens, n_actions),
             trial_input_modifier.reshape(1, n_tokens, n_modifiers)],
             trial_output_action.reshape(1, n_outputs, n_output_actions)
             )
    




def create_model():
    action_input = Input(
            shape=(None, n_actions), 
            name="action_input"
        )
    modifier_input = Input(
            shape=(None, n_modifiers), 
            name="modifier_input"
        )
    
    all_inputs = keras.layers.concatenate(
            [
                    action_input, 
                    modifier_input, 
            ])
    lstm = LSTM(
            units = 50, 
            return_sequences=True,
            name="lstm"
        )(all_inputs)
    output = TimeDistributed(
                Dense(
                    units=n_output_actions, 
                    activation="softmax"                
                ),
                name="output"
            )(lstm)   # add somewhere 10 steps
    
    model = Model(
            inputs = [action_input, modifier_input], 
            outputs = [output]
        )    
    
    return model




def examples_generator():
    while(True):
        yield make_matrix_trial()


model = create_model()
model.compile(
        loss="categorical_crossentropy", 
        optimizer= Adam(lr = 0.001), 
        metrics=['accuracy']#, total_accuracy]
        
        )


#print(model.summary())


model.fit_generator(        
        examples_generator(),
        epochs=100,
        steps_per_epoch=2000,
        verbose=1, 
        validation_data=examples_generator(),
        validation_steps=1000
    )
