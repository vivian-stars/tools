# -*- coding: utf-8 -*-
PATH = r'/home/daviden1013/David_projects/NLP_IE_Pipelines_demo/'

import os

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
tokenizer.save_pretrained(os.path.join(PATH, 'base models', 'PubMedBERT'))

from transformers import AutoModel
base_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
base_model.save_pretrained(os.path.join(PATH, 'base models', 'PubMedBERT'))
