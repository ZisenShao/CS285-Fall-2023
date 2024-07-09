# Lecture 4: Introduction to Reinforcement Learning

- Recap
    
    ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled.png)
    
    - Note:
        - Markov property distinguishes the state from the observation, state satisfy markov property, where observation does not
        - observation is some stochastic function of state which may or may not contain all the information necessary to infer the full state (main difference of observation and state)
    - Recap:
        - Imitation learning: train policy that resemble the expert data
        - RL: train policy without access to expert data
- Reward function
    - $r(s,a)$ is a scalar valued function of the state and the action, sometimes depend on only the state; it tells which states and actions are better
    - Note: The objective in RL is not just to take action have high reward now, but rather to take actions that will lead to higher rewards later - the key of RL is how to take right action now to receive high rewards later
        - Together, the state, the action, the reward, and the transition probabilities define a Markov decision process
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%201.png)
            
        
- Markov chain & Markov decision process
    - Markov chain
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%202.png)
        
        - have $S$ - state space, $T$ - transition operator
    - Markov decision process
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%203.png)
        
        - add two additional objects:
            - $A$ - action space
            - $r$ - reward function: $r:S\times A \rightarrow \mathbb{R}$
                
                ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%204.png)
                
    - Partially observed Markov decision process
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%205.png)
        
        - add two more objects:
            - $O$ - observation space
            - $\varepsilon$ - emission probabiliy: $p(o_t|s_t)$
- The goal of reinforcement learning
    - Process: pass the state into a policy, output the action, then state and action go into the transition probability, which produce the next state
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%206.png)
        
    - Probability distribution over trajectory:
        - trajectory is a sequence of states and actions
        - 3 parts: initial state distribution $p(s_1)$, product over all timestep $\pi_{\theta}(a_t|s_t)$, and probability of the transition to the next timestep $p(s_{t+1}|s_t,a_t)$
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%207.png)
        
    - **Objective of RL:**
        - Goal of RL is to find parameters $\theta$ that define our policy, so as to maximize the expected value of the sum of rewards over the trajectory (the trajectory is produced by the policy)
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%208.png)
        
    - Represent the process as Markov chain:
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%209.png)
        
        - group state and action to form a Markov chain
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2010.png)
        
    - Finite horizon case: state-action marginal
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2011.png)
        
    - Infinite horizon case: stationary distribution
        - consider the objective of RL
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2012.png)
            
        - what if $T$ goes to $\infty$?
            - The objective will become ill-defined, e.g. reward is always positive, and we have a sum of infinite positive number, the reward goes to infinity and cannot be maximized
            - So we need to make objective finite!
        - consider the state-action transition operator
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2013.png)
            
            - We can see, if a state times the operator, and the state hold the same, it converge
        - Imply the stationary (the same state before and after transition)
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2014.png)
            
            - $\mu$ is the state that is stationary
        - We rewrite the objective of RL by dividing infinite $T$, then we get the new well-defined finite objective
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2015.png)
            
    - Expectations interesting point
        - In RL our goal and objective is all about expected value
        - **Expected value interesting point:**
            - expected values can be continuous in the parameters of the corresponding distributions, even when the function that we’re taking the expectation of is itself highly discontinuous
            - This is very important for understanding why RL algorithms can use smooth optimization methods like gradient descent to optimize objectives that are seemingly non-differentiable like binary rewards for winning or losing a game
        - Example:
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2016.png)
            
            - car fall -1, car not fall +1, is binary
            - but the expectation of probability of fall or not is continuous, differentiable
- Algorithm
    - The anatomy of a reinforcement learning algorithm
        - all algorithm mainly follow the same anatomy, which have 3 basic parts
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2017.png)
        
        1. generate samples (i.e. run the policy in the environment)
            - RL is about learning through trial and error, it attempting to run your policy in your environment, interact with your Markov decision process and collect samples (trajectories)
        2. fit a model / estimate the return (like forward)
            - learning a model; estimate something about the current policy
            - i.e. measure how good the trajectories are
        3. improve the policy (like backpropagation)
            - where you actually change your policy to make it better
        - Example:
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2018.png)
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2019.png)
            
    - Which parts are expensive?
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2020.png)
        
        - We have different algorithms cover different box
    - Q-function and Value Functions
        
        ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2021.png)
        
        - Q-function
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2022.png)
            
            - **Expected Total Reward**: The Q-function represents the expected total reward from time step $t$ onward, starting in state $s_t$ and taking action $a_t$, following the policy $\pi_{\theta}$ until the final time step $T$.
        - value function
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2023.png)
            
            - value function is the total reward from $s_t$
            - The relation of Q-function and value function is clear and well-defined
            - The objective is simply the whole reward starting from $s_1$, we can represent it in a way using value function and Q-function
        - The important usage and ideas
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2024.png)
            
        - Q-function and value function is for green box
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2025.png)
            
            - They fundamentally are objects that evaluate how good your policy currently is, so you will fit them or learn them in the green box, and use them in the blue box to improve the policy
    - Types of Algorithms
        - recall the objective
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2026.png)
            
        - Policy gradients: directly differentiate the above objective
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2027.png)
            
        - Value-based: estimate value function or Q-function of the optimal policy (no explicit policy)
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2028.png)
            
        - Actor-critic: estimate value function or Q-function of the current policy, use it to improve policy (usually by gradient)
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2029.png)
            
        - Model-based RL: estimate the transition model, and then…
            - Use it for planning (no explicit policy)
            - Use it to improve a policy
            - Something else
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2030.png)
            
            1. Just use the model to plan (no policy)
                - Trajectory optimization/optimal control (primarily in continuous spaces) - essentially backpropagation to optimize over actions
                - Discrete planning in discrete action spaces – e.g., Monte Carlo tree search
            2. Backpropagate gradients into the policy
                - Requires some tricks to make it work
            3. Use the model to learn a value function
                - Dynamic programming
                - Generate simulated experience for model-free learner
    - Tradeoffs Between Algorithms
        - Why so many RL algorithms?
            - Different tradeoffs
                - Sample efficiency
                    - Sample efficiency = how many samples do we need to get a good policy?
                    - Most important question: is the algorithm off policy?
                        - **Off policy**: able to improve the policy without generating new samples from that policy
                        - **On policy**: each time the policy is changed, even a little bit, we need to generate new samples
                    
                    ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2031.png)
                    
                - Stability & ease of use
                    - Does it converge?
                    - And if it converges, to what?
                    - And does it converge every time?
                    - Why have these questions?
                        - Supervised learning: almost always gradient descent
                        - Reinforcement learning: often not gradient descent
                            - Q-learning: fixed point iteration
                            - Model-based RL: model is not optimized for expected reward
                            - Policy gradient: is gradient descent, but also often the least efficient!
                            - Value function fitting
                                - At best, minimizes error of fit (“Bellman error”)
                                    - Not the same as expected reward
                                - At worst, doesn’t optimize anything
                                    - Many popular deep RL value fitting algorithms are not guaranteed to converge to anything in the nonlinear case
                            - Model-based RL
                                - Model minimizes error of fit
                                    - This will converge
                                - No guarantee that better model = better policy
                            - Policy gradient
                                - The only one that actually performs gradient descent (ascent) on the true objective
            - Different assumptions
                - Stochastic or deterministic?
                - Continuous or discrete?
                - Episodic (finite horizon) or infinite horizon?
                - Common assumption #1: full observability (you observe states rather than observations)
                    - Generally assumed by value function fitting methods
                    - Can be mitigated by adding recurrence
                - Common assumption #2: episodic learning (an agent's interaction with the environment is divided into distinct episodes)
                    - Often assumed by pure policy gradient methods
                    - Assumed by some model-based RL methods
                - Common assumption #3: continuity or smoothness
                    - Assumed by some continuous value function learning methods
                    - Often assumed by some model-based RL methods
            - Different things are easy or hard in different settings
                - Easier to represent the policy?
                - Easier to represent the model?
    - Examples of Algorithms
        - Value function fitting methods
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2028.png)
            
            - Q-learning, DQN
            - Temporal difference learning
            - Fitted value iteration
        - Policy gradient methods
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2027.png)
            
            - REINFORCE
            - Natural policy gradient
            - Trust region policy optimization
        - Actor-critic algorithms
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2029.png)
            
            - Asynchronous advantage actor-critic (A3C)
            - Soft actor-critic (SAC)
            - DDPG
        - Model-based RL algorithms
            
            ![Untitled](Lecture%204%20Introduction%20to%20Reinforcement%20Learning%207247b7a08797403c9969e6af0394a838/Untitled%2030.png)
            
            - Dyna
            - Guided policy search