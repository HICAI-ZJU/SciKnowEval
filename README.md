<div align="center">

<h1 align="center"> SciKnowEval: Evaluating Multi-level Scientific Knowledge of Large Language Models </h1>

<p align="center">
  <a href="https://arxiv.org/abs/2406.09098">📖 Paper</a> •
  <a href="http://www.scimind.ai/sciknoweval/">🌐 Website</a> •
  <a href="https://huggingface.co/datasets/hicai-zju/SciKnowEval">🤗 Dataset</a> •
  <a href="#2">⌚️ Overview</a> •
  <a href="#3">🏹 QuickStart</a> •
  <a href="#4">🏅 Leaderboard</a> •
  <a href="#6">📝 Cite</a>
</p>

<p align=right><b>博学之 ，审问之 ，慎思之 ，明辨之 ，笃行之。</b></p>
<p align=right>—— 《礼记 · 中庸》 <i>Doctrine of the Mean</i></p>

<div align=center><img src="figure/sciknoweval.png" width="80%" height="100%" /></div>

<p></p>
</div>

The <b>Sci</b>entific <b>Know</b>ledge <b>Eval</b>uation (<b>SciKnowEval</b>) benchmark for Large Language Models (LLMs) is inspired by the profound principles outlined in the “<i>Doctrine of the Mean</i>” from ancient Chinese philosophy. This benchmark is designed to assess LLMs based on their proficiency in **Studying Extensively**, **Enquiring Earnestly**, **Thinking Profoundly**, **Discerning Clearly**, and **Practicing Assiduously**. Each of these dimensions offers a unique perspective on evaluating the capabilities of LLMs in handling scientific knowledge.

## 🆕 News

- **\[Sep 2024\]** We released an [evaluation report of OpenAI o1](http://scimind.ai/sciknoweval/o1) with SciKnowEval.

- **\[Sep 2024\]** We have updated the SciKnowEval paper in [arXiv](https://arxiv.org/abs/2406.09098v2).

- **\[Jul 2024\]** We have recently added the Physics and Materials to SciKnowEval. You can access the dataset [here](https://huggingface.co/datasets/hicai-zju/SciKnowEval) and check out the leaderboard [here](http://scimind.ai/sciknoweval).

- **\[Jun 2024\]** We released the SciKnowEval Dataset and Leaderboard for Biology and Chemistry.


## 📌 Table of Contents

- [⌚️ Overview](#2)
- [🏹 QuickStart](#3)
  - [⬇️ Installation](#3.1)
  - [📜 Prepare data](#3.2)
  - [🛒 Prepare models](#3.3)
  - [🚀 Evaluate](#3.4)
- [🏅 Leaderboard](#4)
- [📝 Cite](#6)
- [✨ Acknowledgements](#7)
---

<h2 id="2">⌚️  Overview</h2>

<h3 id="2.1">✡️ Evaluated Abilities</h3>

* 📖 **L1**: **Studying extensively** (i.e., *knowledge memory*). This dimension evaluates the breadth of an LLM’s knowledge across various scientific domains. It measures the model’s ability to remember a wide range of scientific concepts.

* ❓ **L2**: **Enquiring earnestly** (i.e., *knowledge comprehension*). This aspect focuses on the LLM’s capacity for deep enquiry and exploration within scientific contexts, such as analyzing scientific texts, identifying key concepts, and questioning relevant information.

* 💡 **L3**: **Thinking profoundly** (i.e., *knowledge reasoning*). This criterion examines the model’s capacity for critical thinking, logical deduction, numerical calculation, function prediction, and the ability to engage in reflective reasoning to solve problems.

* 🔨 **L4**: **Discerning clearly** (i.e., *knowledge discernment*). This aspect evaluates the LLM’s ability to make correct, secure, and ethical decisions based on scientific knowledge, including assessing the harmfulness and toxicity of information, and understanding the ethical implications and safety concerns related to scientific endeavors.

* 🔬 **L5**: **Practicing assiduously** (i.e., *knowledge application*). The final dimension assesses the LLM’s capability to apply scientific knowledge effectively in real-world scenarios, such as analyzing complex scientific problems and creating innovative solutions.

<h3 id="2.2">🎯 Domains and Tasks</h3>
<div align=center><img src="figure/task.png" width="80%" height="100%" /></div>
<div align=center><img src="figure/task2.png" width="80%" height="100%" /></div>

<h3 id="2.3">📊 Data Stats</h3>

<div align=center><img src="figure/stats.png" width="80%" height="100%" /></div>

<h3 id="2.4">🛠️ Data Construction</h3>

<div align=center><img src="figure/data_collection.png" width="80%" height="100%" /></div>

<h2 id="3">🏹 QuickStart</h2>
<h3 id="3.1">⬇️ Step 1: Installation</h3>

To evaluate LLMs on SciKnowEval, first clone the repository:
```bash
git clone https://github.com/HICAI-ZJU/SciKnowEval.git
cd SciKnowEval
```
Next, set up a conda environment to manage the dependencies:
```bash
conda create -n sciknoweval python=3.10.9
conda activate sciknoweval
```
Then, install the required dependencies:
```bash
pip install -r requirements.txt
```


<h3 id="3.2">📜 Step 2 : Prepare data</h3>

#### Getting Started with SciKnowEval Benchmark

1. **Download the SciKnowEval Benchmark Data**: To begin evaluating language models using the SciKnowEval benchmark, you should first download our dataset. There are two available sources:

    * 🤗 **HuggingFace Dataset Hub**: Access and download the dataset directly from our HuggingFace page: [https://huggingface.co/datasets/hicai-zju/SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval)

    * **Repository Data Folder**: The dataset is organized by level (L1~L5) and task within the `./raw_data/` folder of this repository. You may download parts separately and consolidate them into a single JSON file as needed.

2. **Prepare Your Model’s Predictions**: Utilize the official evaluation script `eval.py` provided in this repository to assess your model. You are required to prepare your model's predictions in the following JSON format, where each entry must preserve all the original attributes (which can be found in the dataset you downloaded) of the data such as question, choices, answerKey, type, domain, level, task, and subtask. Add your model's predicted answer under the "response" field.

Example JSON format for model evaluation:
```json
[
  {
    "question": "What triggers the activation of platelet integrins?",
    "choices": {
      "text": ["White blood cells", "Collagen exposure", "Adrenaline release", "Nutrient absorption"],
      "label": ["A", "B", "C", "D"]
    },
    "answerKey": "B",
    "type": "mcq-4-choices",
    "domain": "Biology",
    "details": {
      "level": "L2",
      "task": "Cellular Function",
      "subtask": "Platelet Activation"
    },
    "response": "B"  // Insert your model's prediction here
  },
  // Additional entries...
]
```

#### ❗Key Points to Remember
- **Preserve All Original Fields**: Ensure each JSON object retains all the original data fields to maintain the integrity of the evaluation.
- **Model Predictions**: Place your model’s predictions in the "response" field of each JSON object.

By following these guidelines, you can effectively use the SciKnowEval benchmark to evaluate the performance of language models across various scientific tasks and levels.


<h3 id="3.3">🛒 Step 3: Prepare models</h3>

**1. For relation extraction tasks, we need to calculate the text similarity with `word2vec` model. We use *GoogleNews-vectors* pretrained model as the default model.**

- Download `GoogleNews-vectors-negative300.bin.gz` from [this link](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) to local.

> The relation extraction evaluation code was initially developed by the [AI4S Cup](https://bohrium.dp.tech/competitions/3793785610?tab=datasets) team, thanks for their great work!🤗

**2. For tasks that use GPT for scoring, we use OpenAI API to assess answers.**

- Please set your OpenAI API key in the `OpenAI_API_KEY` environment variable. Use `export OPENAI_API_KEY="YOUR_API_KEY"` to set the environment variable.

- If you do not set the `OPENAI_API_KEY` environment variable, the evaluation will automatically **skip the tasks that require GPT scoring**.

- 📣 We select `gpt-4o` as the default evaluator !


<h3 id="3.4">🚀 Step 4: Evaluate</h3>

You can run `eval.py` to evaluate your model:

```bash
data_path="your/model/predictions.json"
word2vec_model_path="path/to/GoogleNews-vectors-negative300.bin"
gen_evaluator="gpt-4o" # the correct model name in OpenAI
output_path="path/to/your/output.json"

export OPENAI_API_KEY="YOUR_API_KEY"
python eval.py \
  --data_path $data_path \
  --word2vec_model_path $word2vec_model_path \
  --gen_evaluator $gen_evaluator \
  --output_path $output_path
```
 

<h2 id="4">🏅 Leaderboard</h2>

The latest leaderboards are shown [here](http://scimind.ai/sciknoweval/#overall).

<h2 id="6">📝 Cite</h2>

```
@misc{feng2024sciknoweval,
    title={SciKnowEval: Evaluating Multi-level Scientific Knowledge of Large Language Models},
    author={Kehua Feng and Keyan Ding and Weijie Wang and Xiang Zhuang and Zeyuan Wang and Ming Qin and Yu Zhao and Jianhua Yao and Qiang Zhang and Huajun Chen},
    year={2024},
    eprint={2406.09098},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<h2 id="7"> ✨ Acknowledgements </h2>

Special thanks to the authors of [LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset](https://github.com/OSU-NLP-Group/LLM4Chem), and the organizers of the [AI4S Cup - LLM Challenge](https://bohrium.dp.tech/competitions/3793785610?tab=datasets) for their inspiring work.

The sections evaluating molecular generation in [`evaluation/utils/generation.py`](./evaluation/utils/generation.py), as well as [`evaluation/utils/relation_extraction.py`](./evaluation/utils/relation_extraction.py), are grounded in their research. Grateful for their valuable contributions ☺️!

### Other Related Projects

- [SciEval](https://github.com/OpenDFM/SciEval)
- [SciBench](https://github.com/mandyyyyii/scibench)
- [SciAssess](https://github.com/sci-assess/SciAssess)
