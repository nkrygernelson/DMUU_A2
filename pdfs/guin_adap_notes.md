It seems that there are several apporaches to consider.

Let me reacp what I learnt.

Two main algorithms for the definition of the Value fucntion in a discrete setting:
backwards induction: you satrt at the end where the are no more future moves to make and so the value function is only the reward at that point. Then you iterate backwards. Fuzzy about what this looks like tho.

Value tieration. Updates all of the value functions at every state iteratively and evenutally the value function propagates and the value fucntions convergt to the correct ones.

Now things are different in the continuous setting, before getting into algorithms we note that there are two main steps in the continuous ADP.
1. We learn the Value function.
2. We make a decision such that we maximize the reward and next step value function

In practice stochastic programming is not so differnent form ADP. WE also generate samples for the uncertainty realization.

The three algorithms:
Backwards induction
Value Iteration (online learning), fits while learning and updates this way
And the OiH method which can be combined with the above (not sure)

The idea is that we generate value function regression targets for the approximator to learn. Another idea is that we want to the samples of the value function to be representative of what the the model will see.

OiH is an answer to this need.

Another approach (this one I don't fully understand. It comes from the egenarl ADP slide). It consists of a forward and backwards process where we gerneate samples and iterate forward and generate realistic next sample sates by using the work in progress policy and the work in progress value function.

I am also confused about this apporach because the slides state that we can use both apporaches paired with this forwards-backawards technique.

