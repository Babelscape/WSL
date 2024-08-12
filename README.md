

<div align="center">

# Word Sense Linking: Disambiguating Outside the Sandbox


[![Conference](http://img.shields.io/badge/ACL-2024-4b44ce.svg)](https://2024.aclweb.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FCD21D)](https://huggingface.co/collections/Babelscape/word-sense-linking-66ace2182bc45680964cefcb)

![i](./assets/Sapienza_Babelscape.png)

</div>
<div align="center">
  
</div>

With this work we introduce a new task: **Word Sense Linking (WSL)**. WSL enhances Word Sense Disambiguation by carrying out both candidate identification (new!) and candidate disambiguation. Our Word Sense Linking model is designed to identify and disambiguate spans of text to their most suitable senses from a reference inventory. 

## Installation

Installation from PyPI

```bash
git clone https://github.com/Babelscape/WSL
cd WSL
pip install .
```


## Usage

WSL is composed of two main components: a retriever and a reader.
The retriever is responsible for retrieving relevant senses from a senses inventory (e.g. WordNet),
while the reader is responsible for extracting spans from the input text and link them to the retrieved documents.
WSL can be used with the `from_pretrained` method to load a pre-trained pipeline.

```python
from wsl import WSL
from wsl.inference.data.objects import WSLOutput

wsl_model = WSL.from_pretrained("Babelscape/wsl-base")
WSLOutput = wsl_model("Bus drivers drive busses for a living.")
```

    WSLOutput(
    text='Bus drivers drive busses for a living.',
    tokens=['Bus', 'drivers', 'drive', 'busses', 'for', 'a', 'living', '.'],
    id=0,
    spans=[
        Span(start=0, end=11, label='bus driver: someone who drives a bus', text='Bus drivers'),
        Span(start=12, end=17, label='drive: operate or control a vehicle', text='drive'),
        Span(start=18, end=24, label='bus: a vehicle carrying many passengers; used for public transport', text='busses'),
        Span(start=31, end=37, label='living: the financial means whereby one lives', text='living')
    ],
    candidates=Candidates(
        candidates=[
                    {"text": "bus driver: someone who drives a bus", "id": "bus_driver%1:18:00::", "metadata": {}},
                    {"text": "driver: the operator of a motor vehicle", "id": "driver%1:18:00::", "metadata": {}},
                    {"text": "driver: someone who drives animals that pull a vehicle", "id": "driver%1:18:02::", "metadata": {}},
                    {"text": "bus: a vehicle carrying many passengers; used for public transport", "id": "bus%1:06:00::", "metadata": {}},
                    {"text": "living: the financial means whereby one lives", "id": "living%1:26:00::", "metadata": {}}
        ]
    ),
)



## Model Performance

Here you can find the performances of our model on the [WSL evaluation dataset](https://huggingface.co/datasets/Babelscape/wsl).

### Validation (SE07)

| Models       | P    | R      | F1     |
|--------------|------|--------|--------|
| BEM_SUP      | 67.6 | 40.9   | 51.0   |
| BEM_HEU      | 70.8 | 51.2   | 59.4   |
| ConSeC_SUP   | 76.4 | 46.5   | 57.8   |
| ConSeC_HEU   | **76.7** | 55.4   | 64.3   |
| **Our Model**| 73.8 | **74.9** | **74.4** |

### Test (ALL_FULL)

| Models       | P    | R      | F1     |
|--------------|------|--------|--------|
| BEM_SUP      | 74.8 | 50.7   | 60.4   |
| BEM_HEU      | 76.6 | 61.2   | 68.0   |
| ConSeC_SUP   | 78.9 | 53.1   | 63.5   |
| ConSeC_HEU   | **80.4** | 64.3   | 71.5   |
| **Our Model**| 75.2 | **76.7** | **75.9** |


## Cite this work

If you use any part of this work, please consider citing the paper as follows:

```bibtex
@inproceedings{bejgu-etal-2024-wsl,
    title     = "Word Sense Linking: Disambiguating Outside the Sandbox",
    author    = "Bejgu, Andrei Stefan and Barba, Edoardo and Procopio, Luigi and Fern{\'a}ndez-Castro, Alberte and Navigli, Roberto",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month     = aug,
    year      = "2024",
    address   = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
}
```

## License

The data and software are licensed under cc-by-nc-sa-4.0 you can read it here [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](./wsl_data_license.txt).

