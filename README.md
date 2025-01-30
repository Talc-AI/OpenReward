# OpenReward

OpenReward is a repository of shared reward functions for language modelling tasks.

## Project goals

### Version 0: Proof of concept
- [ ] Basic structure of the project
- [ ] Reward function for formatting
- [ ] Reward function for math problem correctness
- [ ] Ability to integrate with open-r1 project or huggingface TRL grpo demo

### Future versions
 - [ ] Data loading - allow reward functions to be paired with datasets that exercise the reward function
 - [ ] Code syntax reward function
 - [ ] Code correctness reward function
 - [ ] Code speed reward function
 - [ ] Human feedback reward function (offline)
 - [ ] Neural network reward function
 - [ ] Language consistency reward function
 - [ ] Function calling format reward function

 ## Design questions

Q: Where do we decide what reward functions are run for a given generation? Do we decide this at all?
A: For now, just write the reward functions and let the user decide which ones to run.

Q: Is there a chat format we should use for the inputs to the reward function?
A: We should use the messages/chat format from the OpenAI API.

## Compatibility

OpenReward has built in support to integrate nicely with the following projects:
 - [ ] TinyZero
 - [ ] Huggingface TRL GRPO/open-r1