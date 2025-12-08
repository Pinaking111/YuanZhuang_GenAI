"""Simple CLI for Assignment5: train or generate with the fine-tuned model.
"""
import argparse
import os

def train(args):
    # call fine_tune.py programmatically or via subprocess
    import subprocess
    cmd = [
        'python',
        os.path.join(os.path.dirname(__file__), 'fine_tune.py'),
        '--output_dir', args.output_dir,
        '--num_train_epochs', str(args.epochs),
        '--per_device_train_batch_size', str(args.batch_size),
    ]
    subprocess.check_call(cmd)


def generate(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    prompt = args.prompt or "Question: What is AI? Context: AI stands for artificial intelligence.\nAnswer: "
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, top_k=50, top_p=0.95)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('\n--- Generated text ---')
    print(text)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_train = sub.add_parser('train')
    p_train.add_argument('--output_dir', default='assignment5/checkpoint')
    p_train.add_argument('--epochs', type=int, default=1)
    p_train.add_argument('--batch_size', type=int, default=2)

    p_gen = sub.add_parser('generate')
    p_gen.add_argument('--model_dir', default='assignment5/checkpoint')
    p_gen.add_argument('--prompt', default=None)
    p_gen.add_argument('--max_new_tokens', type=int, default=64)

    args = parser.parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'generate':
        generate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
