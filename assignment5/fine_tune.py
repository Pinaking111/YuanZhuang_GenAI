"""Fine-tune GPT-2 (small) on SQuAD QA pairs for a short smoke-test.

This script prepares a prompt of the form:
  Q: <question>\nA: <answer>
and fine-tunes a causal LM (GPT-2) so it learns to continue the prompt with the answer.

The script is configured for a quick smoke-run (small slice of the dataset, 1 epoch,
batch size 1). Adjust `split`, `num_train_epochs` or `per_device_train_batch_size` for larger runs.
"""
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def preprocess_batch(examples, tokenizer, max_length=128):
    prompts = []
    for q, a in zip(examples["question"], examples["answers"]):
        ans_text = a.get("text", [""])[0] if len(a.get("text", [])) > 0 else ""
        prompt = "Q: " + q + "\nA: " + ans_text
        prompts.append(prompt)
    tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
    # for causal LM, labels = input_ids
    tokenized["labels"] = [list(x) for x in tokenized["input_ids"]]
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='openai-community/gpt2')
    parser.add_argument('--output_dir', default='assignment5/checkpoint')
    parser.add_argument('--split', default='train[:100]')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    args = parser.parse_args()

    print('Loading dataset slice', args.split)
    ds = load_dataset('rajpurkar/squad', split=args.split)

    print('Loading tokenizer and model...', args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    print('Tokenizing (batched)...')
    tokenized = ds.map(lambda ex: preprocess_batch(ex, tokenizer, max_length=args.max_length), batched=True, remove_columns=ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_strategy='epoch',
        logging_steps=10,
        fp16=False,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print('Starting training...')
    trainer.train()
    print('Saving model to', args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
"""Minimal fine-tuning script for causal LM (GPT-2) on SQuAD-derived QA pairs.

This script is intentionally minimal and intended as a starting point.
It prepares text by concatenating question and context, then setting the label to be the answer text appended
so the model learns to continue the prompt with the answer.
"""
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def prepare_examples(example, tokenizer, max_length=512):
    # create an input that is: "Question: <question> Context: <context>\nAnswer: "
    prompt = f"Question: {example['question']} Context: {example['context']}\nAnswer: "
    target = example['answers']['text'][0] if len(example['answers']['text'])>0 else ''
    full = prompt + target
    enc = tokenizer(full, truncation=True, max_length=max_length)
    enc['labels'] = enc['input_ids'].copy()
    return enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='openai-community/gpt2')
    parser.add_argument('--output_dir', default='assignment5/checkpoint')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    args = parser.parse_args()

    print('Loading dataset...')
    ds = load_dataset('rajpurkar/squad')

    print('Loading tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # ensure pad token exists for batching
    """Fine-tune GPT-2 (small) on SQuAD QA pairs for a short smoke-test.

    This script prepares a prompt of the form:
      Q: <question>\nA: <answer>
    and fine-tunes a causal LM (GPT-2) so it learns to continue the prompt with the answer.

    The script is configured for a quick smoke-run (small slice of the dataset, 1 epoch,
    batch size 1). Adjust `split`, `num_train_epochs` or `per_device_train_batch_size` for larger runs.
    """
    import argparse
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )


    def preprocess_batch(examples, tokenizer, max_length=128):
        prompts = []
        for q, a in zip(examples["question"], examples["answers"]):
            ans_text = a.get("text", [""])[0] if len(a.get("text", [])) > 0 else ""
            prompt = "Q: " + q + "\nA: " + ans_text
            prompts.append(prompt)
        tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
        # for causal LM, labels = input_ids
        tokenized["labels"] = [list(x) for x in tokenized["input_ids"]]
        return tokenized


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', default='openai-community/gpt2')
        parser.add_argument('--output_dir', default='assignment5/checkpoint')
        parser.add_argument('--split', default='train[:1%]')
        parser.add_argument('--max_length', type=int, default=128)
        parser.add_argument('--num_train_epochs', type=int, default=1)
        parser.add_argument('--per_device_train_batch_size', type=int, default=1)
        args = parser.parse_args()

        print('Loading dataset slice', args.split)
        ds = load_dataset('rajpurkar/squad', split=args.split)

        print('Loading tokenizer and model...', args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.resize_token_embeddings(len(tokenizer))

        print('Tokenizing (batched)...')
        tokenized = ds.map(lambda ex: preprocess_batch(ex, tokenizer, max_length=args.max_length), batched=True, remove_columns=ds.column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            save_strategy='epoch',
            logging_steps=10,
            fp16=False,
            weight_decay=0.01,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        print('Starting training...')
        trainer.train()
        print('Saving model to', args.output_dir)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


    if __name__ == '__main__':
        main()
