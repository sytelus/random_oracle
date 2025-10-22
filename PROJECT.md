# Random Oracle

## Historical Context

Here's the historical context of this project. This project is fork of Verbal ized Sampling method as described in paper in file `verbalized_sampling.pdf` located in the project root. You should read this paper throughly and understand it. The verbalized sampling method enables to sample diversified content from LLM via prompt engineering. The paper describes experiments to measure effectiveness of this method which are important for this project so you must understand these experiments and how to run them.

## Project Idea

The goal of the project is to test a new method, compare it against verbalized sampling and obtain all neccesory experimental data to measure the effectiveness of the new method.

The new method is described as follow:

If prompt ask LLM "Tell me a joke" then it simply generates same joke agaian and again even though LLM knows many great jokes. A part of the reason for lack of diversity in output is because prompt itself doesn't have entropy and therefore forward pass generates same output for same prompt. In the new method, we introduce prompt entropy to elicit entropy in output. For example, our prompt could be as follows:

```text
Imagine you know 10000 of the very best and greatest jokes. All of these jokes are completely different from each other. Tell me a joke number 257.
```

Now by changing the number 257 above to some other number, we can generate many many diversified new jokes.

More general form of above prompt is,

```text
Imagine you know {M} different {XYZ} with {PQR}. All of these are completely different from each other with genuine variety making it a most diverse collection. What is the number {K} from this collection?
```

Here `PQR` may be some desirable attributes of item `XYZ`.


## Project Goal

Based on above project idea we wish to create new implementation for the experiments in the paper to measure the effectiveness of this idea and compare with the idea in the original paper. We should create new files for our experiments and make sure previous experiments can still be run to reproduce the results in the paper. For exisiting files we should make minimal changes so as not to introduce any bugs to reproduce results in the paper.

Your goal is to implement code for doing thee experiments, run them, obtain the results and create a markdown file with report for comparing results side-by-side of results in original paper.

## Suggested approach

You should create new Python environment, install any neccesory dependencies and download any data you might need. When downloading data, save it $DATA_ROOT/random_oracle directory. Then you can implement the code for experiments and run them and collect result and create markdown. Refer to above paper and rest of the codebase if you have questions.
