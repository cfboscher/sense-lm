### Binary Sentence Classification – GPT-4 Teaching Protocol

We provide the protocol used for teaching GPT-4 our binary classification task, as we consider it as a baseline with the objective of validating the relevance of our work, compared to the current capabilities of pre-trained models. We define the task, we provide a set of examples corresponding to our training set, then we ask the model to classify the test set. 


* ***User**:   My task is to determine whether a given sentence contains a reference to smells or olfaction in general. Can you please help me? You must classify each sentence by returning True if it contains a smell reference, and False if not.  I'll provide you a set of examples formatted as follows:*
  * *"ID" – “text”: "class"*
  * *Examples :*
    * *0 -  "Australian Bush, with traces of plant life as if from another world, but often most delicate in odor even in the fragments of them we see in our greenhouses”: True*
    * *1 -  "They tell us that the very common attempt to conform these fair lives into tile or other patterns”: False*




* ***GPT-4**:*
  * *Certainly, I understand your task. You can provide me with sentences, and I'll classify them as either containing a reference to smells or olfaction (True) or not containing such a reference (False). Please don't hesitate to list the sentences you'd like me to analyze.*


  
* ***User**:  First learn from this training set, I will give you sentences to classify afterward.*
  
  * < Train Set >



* ***GPT-4**:  Thank you for providing the examples. These will be helpful for understanding how to classify the sentences you'll give me later. Based on these examples, it looks like I'll need to identify any references to smells, odors, fragrances, or any other olfactory terms in each sentence.*
  

* ***User**: From the classification you have learned, can you classify all these sentences by determining whether they contain a reference to smell or not? Please return a list containing the ID of each sentence and its class (True or False)*


* ***GPT-4**:   Certainly, I can classify these sentences based on whether they contain a reference to smell or not.*
  * *0 – False*
  * *1 – True*
  * *2 – True*
  * *[...]*


