# SENSE-LM

## Description

The repository of ["SENSE-LM : A Synergy of Language Models and Sensorimotor Representations for Sensory Information Extraction"](https://aclanthology.org/2024.findings-eacl.119/).


## Usage
- The [source code](https://github.com/cfboscher/sense-lm/tree/main/code) of SENSE-LM along with instructions to run the code are in ```code```. Please refer to ```code/README.md``` for further instructions.
- The [datasets](https://github.com/cfboscher/sense-lm/tree/main/data) used in the experiments are in ```data```. Please refer to ```data/README.md``` for further instructions.

## Citation

Please cite the original paper in case of use of code or/and dataset. 

*Cédric Boscher, Christine Largeron, Véronique Eglin, and Elöd Egyed-Zsigmond. 2024. SENSE-LM : A Synergy between a Language Model and Sensorimotor Representations for Auditory and Olfactory Information Extraction. In Findings of the Association for Computational Linguistics: EACL 2024, pages 1695–1711, St. Julian’s, Malta. Association for Computational Linguistics.*

```
@inproceedings{boscher-etal-2024-sense,
    title = "{SENSE}-{LM} : A Synergy between a Language Model and Sensorimotor Representations for Auditory and Olfactory Information Extraction",
    author = {Boscher, C{\'e}dric  and
      Largeron, Christine  and
      Eglin, V{\'e}ronique  and
      Egyed-Zsigmond, El{\"o}d},
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.119/",
    pages = "1695--1711",
    abstract = "The five human senses {--} vision, taste, smell, hearing, and touch {--} are key concepts that shape human perception of the world. The extraction of sensory references (i.e., expressions that evoke the presence of a sensory experience) in textual corpus is a challenge of high interest, with many applications in various areas. In this paper, we propose SENSE-LM, an information extraction system tailored for the discovery of sensory references in large collections of textual documents. Based on the novel idea of combining the strength of large language models and linguistic resources such as sensorimotor norms, it addresses the task of sensory information extraction at a coarse-grained (sentence binary classification) and fine-grained (sensory term extraction) level.Our evaluation of SENSE-LM for two sensory functions, Olfaction and Audition, and comparison with state-of-the-art methods emphasize a significant leap forward in automating these complex tasks."
}
```


