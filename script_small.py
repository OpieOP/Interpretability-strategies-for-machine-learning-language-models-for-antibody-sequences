# Some imports 
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import numpy as np
import torch
from datasets import load_dataset,Dataset, DatasetDict
import os
import pandas as pd
from datasets import load_from_disk
# Function to extract embeddings after training
def extract_embeddings_and_save(model, dataloader):
    model.eval()
    embeddings = []
    first_embeddings = []
    attention_scores = []
    model.to("cuda")

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            print(model.device)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions

            embeddings.append(hidden_states[-1].detach().cpu().numpy())
            attention_scores.append(attentions[-1].detach().cpu().numpy())
            first_embeddings.append(hidden_states[0].detach().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    attention_scores = np.concatenate(attention_scores, axis=0)
    first_embeddings = np.concatenate(first_embeddings,axis=0)
    np.save("/home/om423/saves/embeddings_small.npy", embeddings)
    np.save("/home/om423/saves/attentions_small.npy", attention_scores)
    np.save("/home/om423/saves/f_embeddings_small.npy", first_embeddings)


def main():
    # Load dataset
    tokenizer = RobertaTokenizer.from_pretrained("/home/om423/antibody-tokenizer/")

    # Initialize the data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    
    tokenized_dataset = load_from_disk('/home/om423/antibody_tokenized_dataset12')
    print(tokenized_dataset)

    # Display the tokenized dataset

    # # Training arguments
    # n_layer = 12
    # max_memory_per_batch = 7818265*n_layer*6 # consider mixed precision, 4*1.5 bytes for each parameter
    # n = 24*1024**3/max_memory_per_batch
    # batch_s = int(2**(np.floor(np.log2(n))))

    antiberta_config = {
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "hidden_size": 768,
        "d_ff": 3072,
        "vocab_size": 25,
        "max_len": 150,
        "max_position_embeddings": 152,
        "batch_size": 96,
        "max_steps": 225000,
        "weight_decay": 0.01,
        "peak_learning_rate": 0.0001,
        "output_attentions": False,
        "output_hidden_states": False,
    }

    model_config = RobertaConfig(
        vocab_size=antiberta_config.get("vocab_size"),
        hidden_size=antiberta_config.get("hidden_size"),
        max_position_embeddings=antiberta_config.get("max_position_embeddings"),
        num_hidden_layers=antiberta_config.get("num_hidden_layers", 1),
        num_attention_heads=antiberta_config.get("num_attention_heads", 1),
        d_ff = antiberta_config.get("d_ff", 3072),
        type_vocab_size=1,
        output_attentions=False,
        output_hidden_states=False
    )
    
    model = RobertaForMaskedLM(model_config)

    training_args = TrainingArguments(
        output_dir="/home/om423/saves/checkpoints_small",
        overwrite_output_dir=True,
        per_device_train_batch_size=96,
        per_device_eval_batch_size=96,
        max_steps=500000,
        save_steps=2500,
        save_total_limit = 3,
        logging_steps=2500,
        eval_steps=2500,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        warmup_steps=10000,
        learning_rate=1e-4,
        fp16=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  
        greater_is_better=False,
        seed=2024,
        logging_dir='/home/om423/logs/script_small', 
    )
    def verify_device(module, input, output):
        for tensor in input:
            if isinstance(tensor, torch.Tensor):
                assert tensor.device.type == 'cuda', f"Tensor {tensor} is on {tensor.device}, expected cuda"
        for tensor in output:
            if isinstance(tensor, torch.Tensor):
                assert tensor.device.type == 'cuda', f"Tensor {tensor} is on {tensor.device}, expected cuda"

    # Ajouter le hook de vérification à chaque couche du modèle
    
    for module in model.modules():
        module.register_forward_hook(verify_device)
    # Check if GPU is available and print status
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # else:
    #     device = torch.device("cpu")
    #     print("Using CPU")
    # Initialize the trainer

    model.to(torch.device("cuda"))
    print(f"Model is on {next(model.parameters()).device}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.00001)],
    )
    # Check for existing checkpoints in the specified directory
    last_checkpoint = None
    # if os.path.isdir(training_args.output_dir):
    #     checkpoints = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    #     if checkpoints:
    #         last_checkpoint = max(checkpoints, key=os.path.getmtime)

    # # Train the model
    # if len(os.listdir("./test")) != 0:
    #     # check whether there is existing checkpoint
    #     print("Resume training")
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     print("Start training")
    #     trainer.train()

    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Starting training from scratch")
        model.to("cuda")
        print(model.device)
        trainer.train()

    # Create a DataLoader for the evaluation dataset
    # eval_dataloader = trainer.get_test_dataloader(tokenized_dataset["test"])
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    # Extract and save embeddings
    # extract_embeddings_and_save(model, eval_dataloader)
    trainer.save_model("/home/om423/saves/antiberta_small")

if __name__ == "__main__":
    main()
