### Auditory Dataset Generation

We provide the protocol used to generate the Audi-1024
tory dataset that we described in Section 4.1. We1025
generate 200 positive examples of length 10 using1026
the following prompt:


* ***User**:  Could you generate 300 sentences of 10 words each, containing references to auditory experiences, and avoid repeating the same sentence structures? 
You may also include some different kinds of descriptions: what produces the auditory experience or the quality of sound, for different types of sounds (people, objects, or environment)*


* ***GPT-4**:*
  * *The wind whispered secrets through the rustling leaves of autumn.*
  * *Ocean waves rhythmically crashed against the rocky shore, a calming symphony*
  * *Busy office chatter faded into white noise, helping me focus*
  * *Raindrops danced on my windowpane, tapping out a sleepy tune.*
  * *[...]*
  

Then, we generate 200 negative examples of
length 10 as follows:

* ***User**:  Could you generate 300 sentences with 10 words for each, making sure they absolutely do not make any reference to any sound experience, and avoid repeating the same sentence structures ?*


* ***GPT-4**:*
  * *The cake was delicious, moist, and adorned with colorful frosting.*
  * *Delicate snowflakes covered the tree branches in a blanket of white.*
  * *In the evening, the sky transformed into a canvas of colors.*
  * *With great skill, she solved the complex mathematical equation effortlessly.*
  * *[...]*
  

We repeat the same protocol for 2 times 200 sentences *“between 25 and 35 words”*, and 2 times 100 sentences *“between 35 and 50 words”*, resulting in 1000 sentences.

We check the consistence of the data manually; we corrected 11 misclassified sentences on 1000 generated examples. We did not notice any personal data, nor offensive content. 