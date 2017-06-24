# py-pomdp

[![Build Status](https://travis-ci.org/mbforbes/py-pomdp.svg?branch=master)](https://travis-ci.org/mbforbes/py-pomdp)
[![Coverage Status](https://img.shields.io/coveralls/mbforbes/py-pomdp.svg)](https://coveralls.io/r/mbforbes/py-pomdp?branch=master)
[![license MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mbforbes/py-pomdp/blob/master/LICENSE.txt)

A small library to parse POMDP environments and policies, load them into Python
objects, and provide methods for updating beliefs and accessing data.

## Usage

### Loading

The following code shows example usage with the POMDP 'voicemail' (modified 'tiger room') scenario described by Jason Williams in [this paper](http://research.microsoft.com/pubs/160935/williams2007csl.pdf). All relevant files are provided in the `examples/` directory.

```python
import numpy as np
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
pomdp = POMDP(
    'examples/env/voicemail.pomdp',  # env
	'examples/policy/voicemail.policy',  # policy
	np.array([[0.65], [0.35]])  # prior
)

# Let's try some belief updates with the full POMDP.
observations = ['hearDelete', 'hearSave', 'hearSave']
obs_idx = 0
best_action_str = None
while True:
    print 'Round', obs_idx + 1
    best_action_num, expected_reward = pomdp.get_best_action()
    best_action_str = pomdp.get_action_str(best_action_num)
    print '\t- action:         ', best_action_str
    print '\t- expected reward:', expected_reward
    if best_action_str != 'ask':
        # We have a 'terminal' action (either 'doSave' or 'doDelete')
        break
    else:
        # The action is 'ask': Provide our next observation.
        obs_str = observations[obs_idx]
        obs_idx += 1
        print '\t- obs given:      ', obs_str
        obs_num = pomdp.get_obs_num(obs_str)
        pomdp.update_belief(best_action_num, obs_num)
        # Show beliefs
        print '\t- belief:         ', np.round(pomdp.belief.flatten(), 3)

# Take the 'terminal' action, and beliefs should be reset to prior.
best_action_num, expected_reward = pomdp.get_best_action()
pomdp.update_belief(best_action_num,
    pomdp.get_obs_num('hearSave')) # Observation doesn't affect this action.
print '\t- belief:         ', np.round(pomdp.belief.flatten(), 3)
```

The following will be output:
```
Round 1
	- action:          ask
	- expected reward: 3.4619529
	- obs given:       hearDelete
	- belief:          [ 0.347  0.653]
Round 2
	- action:          ask
	- expected reward: 2.91002333333
	- obs given:       hearSave
	- belief:          [ 0.586  0.414]
Round 3
	- action:          ask
	- expected reward: 3.13453841127
	- obs given:       hearSave
	- belief:          [ 0.79  0.21]
Round 4
	- action:          doSave
	- expected reward: 5.14634218527
	- belief:          [ 0.65  0.35]
```

## File specifications

### Environment (`.pomdp`)

For the supported POMDP environment spec, see [Tony's POMDP file format
description](http://cs.brown.edu/research/ai/pomdp/examples/pomdp-file-spec.html)

I built the tool to work with the semantics described on that page, without
consulting the [formal
grammar](http://cs.brown.edu/research/ai/pomdp/examples/pomdp-file-grammar.html).
If anyone uses this tool and discovers it lacks support for syntax they
require, I welcome issues as well as pull requests!

### Policy (`.policy`)

The supported POMDP policy is a list of alpha vectors in XML. For more info, see:

* the file `examples/policy/voicemail.policy` in this repository for an example

* [APPL](http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.HomePage)
  for a solver that outputs this format

* [Tony's Tutorials](http://cs.brown.edu/research/ai/pomdp/tutorial/index.html)
  to learn more about approximate solvers and the policies they produce

## Tests

```bash
# Install tools for linting and code coverage
pip install pep8 coverage

# Lint
pep8 pomdp.py test/tester.py

# Run tests with code coverage
coverage run --source pomdp -m test.tester

# Generate html coverage report; afterwards, point browser to htmlcov/index.html
coverage html
```

## Requirements

* [Numpy](http://www.numpy.org/)  (general)
* [ElementTree](http://effbot.org/zone/element-index.htm)  (for loading policy)

```bash
pip install -r requirements.txt
```

## Resources

* For a great intro to POMDPs, see section 2.1 of [this
  paper](http://research.microsoft.com/pubs/160935/williams2007csl.pdf) by
  Jason Williams.

* For a solver that takes the supported environment format as input, and
  outputs the supported policy format, see
  [APPL](http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.HomePage).

* For lots of information about POMDPs, see [Tony's POMDP
  Page](http://cs.brown.edu/research/ai/pomdp/).
