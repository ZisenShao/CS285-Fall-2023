# Lecture 2: Supervised Learning of Behaviors & Imitation Learning

- Terminology & notation
    
    ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled.png)
    
    - $o_t$: analogy to image in image classification
    - $a_t$: analogy to label in image classification
    - $s_t$: distinct from observation, state encodes all information you need to produce the observation (e.g., state is mass, velocity and direction of animal, observation is the image captured, action is what you decide to do next)
    - $t$: note in the case for RL, usually time are take into account
    - $\pi_\theta (a_t | o_t)$: the policy, $o$ can be replaced by $s$
    - $s_t$ → $o_t$, $o_t$→$a_t$, $s_t$, $a_t$ → $s_{t+1}$,
    - **Note**: A policy needs to provide us with an action to take, in the most general case, policies are distributions meaning that they assign a probability to all the possible actions given a particular observation
    - **Markov Property**: $s_{t+1}$ is conditionally independent of $s_1 ... s_{t-1}$ given $s_t$
    - Note: do not confounding $o$ and $s$, while many times this confusion is benign because the algorithm doesn’t matter it’s state or observation
- Imitation Learning
    - behavioral cloning (basic)
        - Definition and Example:
            
            ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%201.png)
            
            - attempt to clone the behavior of the human demonstrator
            - Example: for automatic driving, human collect data, the camera take picture as observation-feature-input, human record steering angle as action-label-output, and we can train the data using supervised learning to clone human behavior.
            - We develop on this basic
        - How naive behavior cloning can fail, and how we can address the problems
            
            ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%202.png)
            
            - behavioral cloning drawback:
                - It is like supervised learning, but supervised learning assumes the data is i.i.d., but in the sequential setting, the data has dependency and slight mistake could lead to random large mistake in the future
                    
                    ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%203.png)
                    
            - How to make naive behavioral cloning work:
                - Be smart about how we collect (and augment) our data
                    - for automatic driving, use left and right camera to correct the direction
                - Use very powerful models that make very few mistakes
                - Use multi-task learning
                - Change the algorithm (DAgger)
    - Why does behavioral cloning fail? (Math theory analysis)
        - Distributional shift:
            - The distribution under which the policy is tested is shifted from the distribution under which it’s trained
            - The shift is due to the policy’s own mistakes, earlier mistakes will build up future mistakes
            
            ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%204.png)
            
        - Solution to distributional shift - define cost function:
            - We don’t maximize expectation of producing action on observation
                
                ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%205.png)
                
            - instead, we want to minimize the cost between actual action the policy made and human expert action
                
                ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%206.png)
                
                - cost is 0 when the policy action is same as human expert action, 1 otherwise; we want to minimize the expected number of mistakes the policy will make
                - we care about the mistakes the policy make when it drive the car, so we care about the cost in expectation under $p_{\pi_\theta} (s_t)$, under the distribution of states the policy will actually see
                - The distinction between this and the prior one is: we are training the policy to assign high probability to the action under $p_{data}$, but what we really care about is to minimize the number of mistakes under $p_{\pi_\theta}$
                - Note: the professor started mixing up $s$ and $o$
        - Analysis of behavior cloning:
            - With new defined cost function, behavior cloning is still bad, why?
            - assume: $\pi_\theta (a \neq \pi^\star (s)|s) \leq \epsilon$, for all $s \in D_{train}$ meaning the mistakes of action is less than $\epsilon$
            - but every timestep, if you made a mistake, for the next timestep, you will be put in somewhere you are unfamilar with
            - So the expectation of total cost is:
                
                ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%207.png)
                
            - It shows behavior cloning is bounded, and the bound is a bit bad since quadratic mistakes is large with long timestep.
        - More general analysis of behavior cloning:
            
            ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%208.png)
            
            - $p_\theta (s_t)$ is the distribution over states at timestep $t$: sum of probability we made no mistakes and some other distribution
                - probability we made no mistakes
                    - $1-\epsilon$ is the probability we made no mistakes
                    - so far $t$ timestep, so $(1-\epsilon)^t$
                    - if you don’t make mistakes, you still in distribution $p_{train}$
                - some other distribution
                    - the mistake distribution times 1 minus the term we defined for “probability we made no mistakes”
            
            ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%209.png)
            
            - total variation divergence between $p_\theta (s_t)$ and $p_{train} (s_t)$ is $2 \epsilon t$
                - 2 is because the worst case is that in one state one of the probability is 1, the other is 0 and in some other state it’s the other way around; so the worst possible difference between 2 distributions when you sum over all the states is 2
            - Based on the total variation divergence, we derive the bound of expectation value of costs is $\epsilon T + 2\epsilon T^2$, which is $O(\epsilon T^2)$
        - The analysis result shows that the bound of the expectation costs of behavior cloning is $O(\epsilon T^2)$, it is quadratic bound
        - How do we make behavior cloning work?
            1. teach the policy to recover from mistakes. For example, car left and right camera tells the policy what it will see when it makes mistakes, so it corrects mistakes
            2. make the dataset have more mistakes, i.e., has a broader training distribution. For example, if we make more training trajectories with some small mistakes and then recover from those mistakes, then if the policy makes small mistakes, it is still in distribution and could correct the mistakes
            - A paradox: imitation learning can work better if the data has more mistakes (and recoveries)!
            
            ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2010.png)
            
    - Addressing the problem of behavior cloning in practice
        
        ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2011.png)
        
        - Be smart about how we collect (and augment) our data
            - Intentionally add mistakes and corrections
                - The mistakes hurt, but the corrections help, often more than the mistakes hurt
            - Use data augmentation
                - Add some “fake” data that illustrates corrections (e.g., side-facing cameras)
        - Use very powerful models that make very few mistakes
            - Why might we fail to fit the expert?
                - Non-Markovian behavior (sol: use sequence model)
                    - The model make decision only based on current state, ignoring the context/history
                    - But human make decision based on whole context/history
                    
                    ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2012.png)
                    
                    - How to use the whole history?
                        - use sequence model like LSTM, Transformer, RNN, etc.
                        - Aside: Why might this work poorly?
                            - it might exacerbate correlations that occur in your data
                            - it might caused “causal confusion”, see: de Haan et al., “Causal Confusion in Imitation Learning”
                            - Question 1: Does including history mitigate causal confusion?
                            - Question 2: Can DAgger mitigate causal confusion?
                - Multimodal behavior (sol: mixture gaussian, VAE, diffusion, Discretization)
                    - Example: if you come across a tree, go from left or right are both valid and in expert training data, but if you average these out, you will go in middle and hit the tree
                    - Solution:
                        - More expressive continuous distributions
                            - mixture of Gaussians
                                - describes a set of means, covariances and wights
                                - drawbacks: number of modes is chosen by yourself, fixed
                            - latent variable models
                                
                                ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2013.png)
                                
                                - input latent value into the NN, output different mode
                                - drawbacks: choose number randomly during training will make the output arbitrary, should choose in a smart way to correlate with which mode you want to output
                                - The most widely used type of model of this sort is the (conditional) variational autoencoder; it solve the above drawback
                            - diffusion models
                                - analogy action as image, teach model to denoise from action, so model can learn to produce good action from bad random action
                                
                                ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2014.png)
                                
                        - Discretization with high-dimensional action spaces
                            - discretize one dimension at a time
                                
                                ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2015.png)
                                
        - Use multi-task learning
            
            ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2016.png)
            
            ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2017.png)
            
            - In theory, we actually will see distributional shift in two places
        - Change the algorithm (DAgger)
            - Remind:
                - $p_{data}$ is the distribution of states under which the policy is trained
                - $p_{\pi_\theta}$ is the distribution of states under which the policy is tested
            - The intuition of DAgger:
                - Before: we try to change the policy such that $p_{\pi_\theta}$ is close to $p_{data}$ by making fewer mistakes
                - Now: can we change $p_{data}$ so it better covers the states that the policy actually visits, i.e. can we make $p_{data}(o_t) = p_{\pi_\theta}(o_t)$
                - idea: instead of being clever about $p_{\pi_\theta}(o_t)$, be clever about $p_{data}(o_t)$!
            - DAgger: Dataset Aggregation
                - run the policy in the real world, see which states it visits and ask humans to label those states
                
                ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2018.png)
                
                - problem: step 3 is problematic and counterfactual
    - Problem of imitation learning
        - Humans need to provide data, which is typically finite
            - Deep learning works best when data is plentiful
        - Humans are not good at providing some kinds of actions
        - Humans can learn autonomously; can our machines do the same?
            - Unlimited data from own experience
            - Continuous self-improvement
- New Terminology & notation for reward/cost function
    
    ![Untitled](Lecture%202%20Supervised%20Learning%20of%20Behaviors%20&%20Imita%2087c7b325e13a4c898cc40d07482cec76/Untitled%2019.png)
    
    - $r(s,a)=-c(s,a)$