# SENSE-LM 
The data used in "SENSE-LM : A Synergy of Language Models and Sensorimotor Representations for Sensory Information Extraction".

We include three subfolders : 

- `Auditory`, corresponding to the Auditory Dataset described in our paper and annotated by our team.
- `Odeuropa`, corresponding to the Odeuropa English Benchmark Dataset. Please refer to the appropriate section to download the dataset.
- `preprocessed`, storing pre-processed versions of the two aforementionned datasets for facilitating experiments and avoiding repeating the preprocessing steps on each run.
## Auditory Dataset
We built an artificial dataset composed of synthetic  sentences generated with [GPT-4](https://openai.com/blog/chatgpt),

It contains 1002 sentences, with 502 sentences containing references to sound experiences (positives) and 500 negatives.
For each positive sentence, we labelled the terms referring to sound experiences with [Label Studio](https://labelstud.io/).

All sentences are included in the file `Auditory/auditory_dataset_full.csv`, positive sentences are as well included in `Auditory/auditory_dataset_positives.csv`.

Files are structured as follows : 

| ID | text                                                                                       | contains_ref | label                                                                |
|----|--------------------------------------------------------------------------------------------|--------------|----------------------------------------------------------------------|
| 1  | On the architect's drafting table, blueprints held the promise of structures yet to exist. | False        | []                                                                   |
| 2  | The zip of a tent closed out the nocturnal world.                                          | True         | [{"start":0,"end":17,"text":"The zip of a tent","labels":["SOUND"]}] |
| ... | ...                                                                                        | ...          | ...                                                                  |

With the following attributes : 
- ID : sentence ID
- text : the sentence text
- contains_ref : True or False, indicates whether the sentence includes an auditory reference (positive) or not (negative)
- label : List of terms labelled as evoking a sound experience. `start` and `end` mark the characters indices, `text`contains the corresponding text, and `labels` indicate the current label applied (in this case, a unique label `SOUND` is used for all terms.)
### Generation and Annotation Protocol

The prompts used for the dataset creation are provided [here](https://github.com/cfboscher/sense-lm/blob/main/gpt4_prompts/generate_dataset.md).

We carefully ask GPT-4 to create examples respecting a realistic diversity of sentence structures with different sentence lengths (400 sentences of maximum 10 words, 400 sentences of between 25 and 35 words, and 200 sentences between 35 and 50 words) with a ratio of positive sentences examples of 0.5, as detailed in the appendices of our paper.
We check the consistence of 
the data manually; we corrected 11 misclassified sentences on 1000 generated examples. We did not notice any personal data, nor offensive content.


## Odeuropa English Benchmark
This state-of-the-art dataset focused on olfactory experiences from the 15th to the 20th century. We focus on the English Benchmark, containing 2176 sentences with a positive sentence ratio of 0.28 and, 5530 utterances of smell related terms, distributed in 602 sentences.

We use the public dataset published by [1], available on the following repository : https://github.com/Odeuropa/benchmarks_and_corpora

### Downloading instructions
We do not include the dataset to the repository by default. Please download the dataset by running the following command : 
``` 
bash download_data.sh
```

### References
[1] _Tonelli, Sara and Menini, Stefano. FrameNet-like Annotation of Olfactory Information in Texts. In Proceedings of LaTeCH-CLfL 2021_
