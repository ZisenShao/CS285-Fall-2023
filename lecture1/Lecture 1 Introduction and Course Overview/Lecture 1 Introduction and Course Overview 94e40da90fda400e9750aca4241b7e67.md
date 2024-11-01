# Lecture 1: Introduction and Course Overview

- Background & Motivation
    - Introduction Case: Robot arm picking up a ball
        - Real practice is more difficult than directly picking up things in the camera
        - Traditional ML: Supervised learning needs a lot of datasets, such feature-label datasets are limited and not suitable for tasks that need to interact with the environment
        - In RL - essence: let the agent(robot) automatically try different trials and label with success and failure, as the dataset (machine collects data by itself)
            
            ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled.png)
            
    - History
        
        ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%201.png)
        
    - Importance of RL
        - Supervised learning rely on the data, machine can only approximate human behavior
        - RL try to let agent find best solution to task, which can go beyond human behavior
        
        ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%202.png)
        
- Course Content
    1. From supervised learning to decision making
    2. Model-free algorithms: Q-learning, policy gradients, actor-critic
    3. Model-based algorithms: planning, sequence models, etc.
    4. Exploration
    5. Offline reinforcement learning
    6. Inverse reinforcement learning
    7. Advanced topics, research talks, and invited lectures
- What is RL?
    - Definition:
        - Mathematical formalism for learning-based decision making
        - Approach for learning decision making and control from experience
    - Difference from other ML topics
        - i.i.d.: independent and identically distributed data
        - RL doesn’t have a ground truth label, the data is a series of events that you can say explicitly which exact step decides the result. RL is very dependent on and interactive with the environment
            
            ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%203.png)
            
        - RL procedure is cyclical, agent interacts with the world, choose action $A_t$, and the world response with the resulting (consequenses) state (observations) $S_{t+1}$ and reward signal $R_{t+1}$, the reward simply indicates how good that state is, but it doesn’t tell you if the action that you just took was a good or bad action (since it is a time series, the success state $S_{t+1}$ may caused by the good action $A_t$, or something good earlier)
        - data is not given like in supervised learning, in RL, pick your own action and collect your own data
        - goal is to learn a policy $\pi_{\theta}$ which map state to action, the best policy is to maximize the accumulate total reward (not the reward at a timestep), so maybe you can do something not get high reward now, but to get high reward later
            
            ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%204.png)
            
- Examples of problems and applications
    
    ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%205.png)
    
    - Game, robots, transportation, image generation, chip design
- Why should we study deep reinforcement learning?
    - Insights and philosophy
        - Recall that RL discovers new solutions, beyond human
            
            ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%206.png)
            
        - Recommend reading: A bitter lesson
            
            ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%207.png)
            
        - ML is for decision, Brain is for movement, but I think movement is a series of decision making
            
            ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%208.png)
            
        - Why deep model?
            
            ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%209.png)
            
    - Why deep RL?
        - Deep = scalable learning from large, complex datasets
        - Reinforcement learning = optimization
        
        ![Untitled](Lecture%201%20Introduction%20and%20Course%20Overview%2094e40da90fda400e9750aca4241b7e67/Untitled%2010.png)
        
        - Deep is for learning, RL is for searching
- What other problems do we need to solve to enable real-world sequential decision making?
    - Beyond learning from reward
        - Basic reinforcement learning deals with maximizing rewards
        - This is not the only problem that matters for sequential decision making!
        - We will cover more advanced topics
            - Learning reward functions from example (inverse reinforcement learning)
            - Transferring knowledge between domains (transfer learning, meta-learning)
            - Learning to predict and using prediction to act
    - Are there other forms of supervision?
        - Learning from demonstrations
            - Directly copying observed behavior
            - Inferring rewards from observed behavior (inverse reinforcement learning)
        - Learning from observing the world
            - Learning to predict
            - Unsupervised learning
        - Learning from other tasks
            - Transfer learning
            - Meta-learning: learning to learn
- How do we build intelligent machines?
    - Hypothesis: Learning as the basis of intelligence
        - Some things we can all do (e.g. walking)
        - Some things we can only learn (e.g. driving a car)
        - We can learn a huge variety of things, including very difficult things
        - Therefore our learning mechanism(s) are likely powerful enough to do everything we associate with intelligence
        - But it may still be very convenient to “hard-code” a few really important bits
    - Hypothesis: there is a single learning procedure/algorithm that underlies all that we associate with intelligent behavior - i.e., The way to learn anything share a same algorithm/procedure
        - An algorithm for each “module”? Or a single flexible algorithm?
- What challenges still remain?
    - We have great methods that can learn from huge amounts of data
    - We have great optimization methods for RL
    - We don’t (yet) have amazing methods that both use data and RL
    - Humans can learn incredibly quickly, deep RL methods are usually slow
    - Humans reuse past knowledge, transfer learning in RL is an open problem
    - Not clear what the reward function should be
    - Not clear what the role of prediction should be