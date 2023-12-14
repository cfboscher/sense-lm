### Sensory terms Extraction – GPT-4 Teaching Protocol

We  provide the protocol used for teaching GPT-4 our sensory words extraction tasks. We define the task, we provide a set of examples corresponding to our training set, then we ask the model to classify the test set

* ***User**:   I need you to extract words in the following sentences, that evoke smells: it can be words explicitly evoking smells, words describing smell qualities, or entities that produce a smell. For instance, in the sentence: “Musk pots generally moist exhales disagreeable predominant ammoniacal smell whenever pots are opened”, you would extract the words “disagreeable, predominant, ammoniacal, musk, smell”. Does it seem fine to you?*




* **GPT-4**: *Yes, I understand the task. You want me to extract words from sentences that relate to smells in various ways—either directly describing a smell, mentioning a quality of a smell, or referring to an entity that produces a smell. I'll give it a try. Please provide the sentences from which you'd like me to extract these words.*
  


* ***User**: I will first provide you a training set of examples.*
  
  * In the sentence < sentence >, you would extract the words < words >
  * < Train Set >



* ***GPT-4**:  Thank you for providing these examples. They give a clear picture of the kind of words you're interested in identifying: terms that relate to smells, odors, and olfactory experiences, whether they are descriptive, noun-based, or rooted in specific substances that are known for their smell.  Now that I've seen these examples, I'm ready to identify olfactory-related words in new sentences you provide. Please don't hesitate to go ahead and share them.*
  

* ***User**: Now I'll provide a list of sentences, I just want you to return, for each, the list of olfactory words you detect in each sentence*
  * < Test Set >


* ***GPT-4**:*
  * *“smelling, eglantine, gilloflowers”*
  * *“hearbe, flower, sent, herbes, sweetest”*
  * *“scent”*
  * *“scent, musk, ambergriese”*
  * *“scents, vertue, odour”*
  * *[...]*


