import logging
from pathlib import Path

import csv
from builtins import str

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import MultiLabelTextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

import OpenAttack as oa

def pubmed_qa():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="bio_text_relation_classification")

    ##########################
    ########## Settings
    ##########################
	
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 10
    batch_size = 32

    evaluate_every = 500
    lang_model = Path("BioMedRoBERTa-PubMedQA")
    do_lower_case = False

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        tokenizer_class="BertTokenizer",
        do_lower_case=do_lower_case)

    ##########################
    ########## Data processor
    ##########################

    label_list = ["yes","no","maybe"]
    metric = "acc"

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=128,
                                            data_dir="PubMedQA-Data",
                                            label_list=label_list,
                                            label_column_name="label",
                                            metric=metric,
                                            quote_char="'",
                                            multilabel=True,
                                            train_filename="Train.tsv",
                                            dev_filename=None,
                                            test_filename="Test.tsv",
                                            dev_split=0,
                                            )

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    ##########################
    ########## Language model
    ##########################

    language_model = LanguageModel.load(lang_model)
    prediction_head = MultiLabelTextClassificationHead(num_labels=len(label_list))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)
		
    
    input_address = 'Dataset\\PubMedQA.tsv'
    basic_texts = []
    
	dataset = []
	
    with open(input_address) as input_file:
        input_data = csv.reader(input_file, delimiter='\t')
        line_num = 0
        for row in input_data:
            if (line_num > 0):
                dataset.append(row)
				
            line_num += 1
			
	victim = oa.DataManager.loadVictim(model, dataset)
	attacker = oa.attackers.HotFlipAttacker()
	attack_eval = OpenAttack.AttackEval(attacker, victim)
	print("After-attack test results (HotFlip):")
	attack_eval.eval(dataset, visualize=True)
	
	
	victim = oa.DataManager.loadVictim(model, dataset)
	attacker = oa.attackers.DeepWordBugAttacker()
	attack_eval = OpenAttack.AttackEval(attacker, victim)
	print("After-attack test results (DeepWordBug):")
	attack_eval.eval(dataset, visualize=True)
	
	
	victim = oa.DataManager.loadVictim(model, dataset)
	attacker = oa.attackers.TextBuggerAttacker()
	attack_eval = OpenAttack.AttackEval(attacker, victim)
	print("After-attack test results (TextBugger):")
	attack_eval.eval(dataset, visualize=True)
	
	
	victim = oa.DataManager.loadVictim(model, dataset)
	attacker = oa.attackers.TextFoolerAttacker()
	attack_eval = OpenAttack.AttackEval(attacker, victim)
	print("After-attack test results (TextFooler):")
	attack_eval.eval(dataset, visualize=True)
    


if __name__ == "__main__":
    pubmed_qa()
