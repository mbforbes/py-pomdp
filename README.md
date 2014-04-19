# py-pomdp

A small tool to parse POMDP environments and policies, load them into Python objects, and provide methods for belief updates and accessing data.

## Usage

### Loading

The following code shows example usage with the POMDP 'Voicemail' (modified tiger room) scenario described by Jason Williams in [this paper](http://research.microsoft.com/pubs/160935/williams2007csl.pdf). All relevant files are provided in the `examples/` directory.

```python
from pomdp import *

# Filenames.
filename_env = 'examples/env/voicemail.pomdp'
filename_policy = 'examples/policy/voicemail.policy'

# Option 1: Load just environment.
env = POMDPEnvironment(filename_env)

# Option 2: Load just the policy.
policy = POMDPPolicy(filename_policy)

# Option 3: Load 'full POMDP' using env, policy, and belief prior.
pomdp = POMDP(filename_env, filename_policy, np.array([[0.65], [0.35]]))
```

### Using

We continue with Option 3 above and run through the sequence shown in [Williams' paper](http://research.microsoft.com/pubs/160935/williams2007csl.pdf) (page 7 of the PDF, page numbered 399). The values output match those expected.

```python
import numpy as np
from pomdp import *

# Load 'full POMDP' using env, policy, and belief prior.
pomdp = POMDP('examples/env/voicemail.pomdp', # env
	'examples/policy/voicemail.policy', # policy
	np.array([[0.65], [0.35]])) # prior

# Let's try some belief updates with the full POMDP.
observations = ['hearDelete', 'hearSave', 'hearSave']
obs_idx = 0
best_action_str = None
while True:
    print 'Round', obs_idx
    best_action_num, expected_reward = pomdp.get_best_action()
    best_action_str = pomdp.get_action_str(best_action_num)
    print '\t- action:         ', best_action_str
    print '\t- expected reward:', expected_reward
    if best_action_str != 'ask':
        # We have a 'terminal' action (either 'save' or 'delete')
        break
    else:
        # The action is 'ask': Provide our next observation.
        obs_str = observations[obs_idx]
        obs_idx += 1
        obs_num = pomdp.get_obs_num(obs_str)
        pomdp.update_belief(best_action_num, obs_num)
        # Show beliefs
        print '\t- belief:         ', list(pomdp.belief)

# Take the 'terminal' action, and beliefs should be reset to prior.
best_action_num, expected_reward = pomdp.get_best_action()
pomdp.update_belief(best_action_num,
    pomdp.get_obs_num('hearSave')) # Observation doesn't affect this action.
print '\t- belief:         ', list(pomdp.belief)
```

The following will be output:
```
Round 0
    - action:          ask
    - expected reward: 3.4619529
    - belief:          [array([ 0.34666667]), array([ 0.65333333])]
Round 1
    - action:          ask
    - expected reward: 2.91002333333
    - belief:          [array([ 0.58591549]), array([ 0.41408451])]
Round 2
    - action:          ask
    - expected reward: 3.13453841127
    - belief:          [array([ 0.79049881]), array([ 0.20950119])]
Round 3
    - action:          doSave
    - expected reward: 5.14634218527
    - belief:          [array([ 0.65]), array([ 0.35])]
```

## File specifications

### Environment (`.pomdp`)
For the supported POMDP environment spec, see [Tony's POMDP file format description](http://cs.brown.edu/research/ai/pomdp/examples/pomdp-file-spec.html)

I built the tool to work with the semantics described on that page, without consulting the [formal grammar](http://cs.brown.edu/research/ai/pomdp/examples/pomdp-file-grammar.html). If anyone uses this tool and discovers it lacks support for syntax they require, I welcome issues as well as pull requests!

### Policy (`.policy`)

The supported POMDP policy is a list of alpha vectors; see APPL for more info (it outputs this format), the file in this repository `examples/policy/voicemail.policy` for an example, and [Tony's Tutorials](http://cs.brown.edu/research/ai/pomdp/tutorial/index.html) to learn what's going on.

## Tests
You can run the environment parser test (loads from `examples/env/env_parser_test.pomdp`) as follows:
```bash
$ python pomdp.py
```

## Requirements
* [ElementTree](http://effbot.org/zone/element-index.htm)
* [Numpy](http://www.numpy.org/)

## Resources
* For a great intro to POMDPs, see section 2.1 of [this paper](http://research.microsoft.com/pubs/160935/williams2007csl.pdf) by Jason Williams.
* For a solver that takes the supported environment format as input, and outputs the supported policy format, see [APPL](http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.HomePage).
* For lots of information about POMDPs, see [Tony's POMDP Page](http://cs.brown.edu/research/ai/pomdp/).
