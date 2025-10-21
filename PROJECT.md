# Random Oracle

## Historical Context

Here's the historical context of this project. This project is fork of Verbal ized Sampling method as described in paper in file `verbalized_sampling.pdf` located in the project root. You should read this paper throughly and understand it. The verbalized sampling method enables to sample diversified content from LLM via prompt engineering. The paper describes experiments to measure effectiveness of this method which are important for this project so you must understand these experiments and how to run them.

## Goal

The goal of the project is to test a new method, compare it against verbalized sampling and obtain all neccesory experimental data to measure the effectiveness of the new method.

The new method is described as follow:

If prompt ask LLM "Tell me a joke" then it simply generates same joke agaian and again even though LLM knows many great jokes. A part of the reason for lack of diversity in output is because prompt itself doesn't have entropy and therefore forward pass generates same output for same prompt. In the new method, we introduce prompt entropy to elicit entropy in output. For example, our prompt could be as follows:

```text
Imagine you know 10000 of the very best and greatest jokes. All of these jokes are completely different from each other. Tell me a joke number 257.
```

Now by changing the number 257 above to something else, we can generate many many diversified new jokes.